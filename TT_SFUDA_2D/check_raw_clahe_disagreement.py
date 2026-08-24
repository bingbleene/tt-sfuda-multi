"""
check_raw_clahe_disagreement.py

MUC DICH (Tang 0 — chua train gi ca)
-------------------------------------
Do disagreement giua source model DONG BANG khi cho an raw view vs CLAHE
view cua CUNG mot anh target khong nhan, o dung resolution da freeze cho
tung domain shift. Day la buoc loc re tien TRUOC KHI quyet dinh co dang
xay Appearance-Diverse Dual Teacher hay khong.

RANH GIOI QUAN TRONG
---------------------
- KHONG doc target mask/label o bat ky dong code nao trong file nay.
  (Neu ban can doi chieu voi Frangi hay bat ky nguon nao khac, lam o
  MOT script rieng, khong trong file nay, de giu ranh gioi ro rang.)
- Day la Tang 0 (frozen-model disagreement). No KHONG chung minh divergence
  song sot qua qua trinh training — muon biet dieu do can Tang 2 (control
  training ngan 3-5 epoch, theo doi D_i qua tung epoch), lam sau, o script
  khac, chi neu Tang 0 pass.

CACH DOC KET QUA (khong dung threshold cung)
---------------------------------------------
1) D_fg_union co lon ro ret hon Dual-EMA disagreement da do truoc day
   (~0.002-0.003) khong?
2) clskeleton_overlap co THAP (divergence nam o CAU TRUC vessel) hay
   D_fg_union lon nhung clskeleton_overlap van cao (nghia la 2 view chi
   lech o do day/nguong, khong lech o cau truc)?
3) fg_ratio_change co duong manh va nhat quan (CLAHE lam foreground phinh
   het cac shift) -> canh bao day co the la artefact tuong phan gia, chua
   chac la diversity huu ich?
4) Pattern (D_fg_union, clskeleton_overlap, fg_ratio_change) co KHAC NHAU
   giua 4 shift, hay collapse thanh cung 1 hanh vi nhu Dual-EMA da tung
   collapse?

CHAY (qua launcher hien tai cua repo, dung quy uoc):
    python run.py check_raw_clahe_disagreement.py \
        --domain_shifts chase2hrf chase2rite hrf2chase hrf2rite \
        --out_dir /kaggle/working/outputs/raw_clahe_diagnostic

DA NOI theo dung unsupervised_selector_v5b.py (nguon xac nhan, khong con
doan mo hinh nua):
    - preprocess: normalize_like_dataset() tu patch_inference.py
    - model output: logit -> torch.sigmoid() (MODEL_OUTPUT_IS_LOGITS=True)
    - checkpoint: archs.__dict__[cfg['arch']](...) + models/{source}/model.pth
    - config: models/{source}/config_{target}_dualema.yml
    - anh target khong nhan: inputs/{target}/train/images, duoi file tu
      cfg['img_ext'] (giu dung ranh gioi da co: train/images = unlabeled
      selection data, test/images+masks chi dung cho AUDIT rieng biet)

CHAY trong TT_SFUDA_2D/ (dung cwd nhu unsupervised_selector_v5b.py):
    python run.py check_raw_clahe_disagreement.py \\
        --domain_shifts chase2hrf chase2rite hrf2chase hrf2rite
"""

import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import cv2
import yaml
import torch
from skimage.morphology import skeletonize

import archs
from patch_inference import normalize_like_dataset

# ============================================================
# CONFIG
# ============================================================

# Resolution da freeze cho tung domain shift (khop voi cau hinh final).
FROZEN_RESOLUTION = {
    "chase2hrf": 1024,
    "chase2rite": 768,
    "hrf2chase": 384,
    "hrf2rite": 512,
}

# CLAHE params — PHAI KHOP voi ClaheLAB dang dung trong repo
# (clahe_transform.py). Neu repo dung clip_limit/tile khac, sua o day.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Xac nhan tu unsupervised_selector_v5b.py: predict() luon ap torch.sigmoid
# len logit tra ve tu model. Giu = True, khong doi tru khi kien truc doi.
MODEL_OUTPUT_IS_LOGITS = True

PRED_THRESHOLD = 0.5
EPS = 1e-7

# Doi chieu tham khao — disagreement Dual-EMA da do truoc day, chi de in
# ra man hinh cho de so sanh, KHONG dung lam threshold quyet dinh cung.
DUAL_EMA_DISAGREEMENT_REFERENCE = (0.002, 0.003)


# ============================================================
# INTEGRATION — noi dung theo unsupervised_selector_v5b.py (da xac nhan)
# ============================================================

def _source_target(domain_shift: str):
    """'chase2hrf' -> ('chase_unet', 'hrf')."""
    src, tgt = domain_shift.split("2")
    return f"{src}_unet", tgt


def load_config(domain_shift: str) -> dict:
    """
    Doc config_{target}_dualema.yml cua source tuong ung — dung file nay
    (khong phai config_{target}.yml goc) vi day la file duy nhat chac
    chan ton tai cho moi shift theo quy uoc sinh config Dual-EMA hien co.
    """
    source, target = _source_target(domain_shift)
    cfg_path = os.path.join("models", source, f"config_{target}_dualema.yml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Khong thay {cfg_path}. Da chay cell sinh config Dual-EMA chua?"
        )
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_frozen_source_model(domain_shift: str, device: str = "cuda"):
    source, _ = _source_target(domain_shift)
    cfg = load_config(domain_shift)

    model = archs.__dict__[cfg["arch"]](
        cfg["num_classes"], cfg["input_channels"], cfg["deep_supervision"]
    ).to(device)

    ckpt_path = os.path.join("models", source, "model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Khong thay checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)  # strict=True mac dinh — bao loi som neu kien truc lech
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_target_image_paths(domain_shift: str):
    """
    Anh target KHONG NHAN — dung 'train/images' (giong dung ranh gioi
    unsupervised_selector_v5b.py: selector chi dung train/images, con
    test/images+masks chi danh cho AUDIT rieng, khong dung o day).
    """
    _, target = _source_target(domain_shift)
    cfg = load_config(domain_shift)
    img_dir = os.path.join("inputs", target, "train", "images")
    paths = sorted(glob(os.path.join(img_dir, "*" + cfg["img_ext"])))
    if not paths:
        raise RuntimeError(f"Khong tim thay anh trong {img_dir}")
    return paths


def preprocess_for_model(img_bgr: np.ndarray, resolution: int) -> torch.Tensor:
    """Khop dung preprocess_tensor() trong unsupervised_selector_v5b.py."""
    img_resized = cv2.resize(
        img_bgr, (resolution, resolution), interpolation=cv2.INTER_LINEAR
    )
    img_norm = normalize_like_dataset(img_resized)
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {path}")
    return img  # BGR, uint8


# ============================================================
# CLAHE view (kenh L trong khong gian LAB)
# ============================================================

def apply_clahe_lab(
    img_bgr: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size=CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ============================================================
# Inference — tra ve prob map [0,1], shape (H, W), float32
# ============================================================

@torch.no_grad()
def predict_prob(model, img_bgr: np.ndarray, resolution: int, device: str) -> np.ndarray:
    x = preprocess_for_model(img_bgr, resolution).to(device)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    prob = torch.sigmoid(out) if MODEL_OUTPUT_IS_LOGITS else out
    prob = prob.squeeze().detach().cpu().numpy().astype(np.float32)
    return prob


# ============================================================
# Metrics — tinh hoan toan tu 2 prob map, KHONG dung nhan that
# ============================================================

def clskeleton_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    clDice-style structural overlap giua 2 mask nhi phan:
        trung binh cua (skeleton_a nam trong mask_b) va
                       (skeleton_b nam trong mask_a).
    1.0 = cau truc trung khop hoan toan; 0.0 = khong lien quan cau truc.
    Day la chi so tach "khac nhau ve CAU TRUC" khoi "khac nhau ve DO DAY".
    """
    if mask_a.sum() == 0 and mask_b.sum() == 0:
        return 1.0
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return 0.0
    skel_a = skeletonize(mask_a)
    skel_b = skeletonize(mask_b)
    t_prec = (skel_a & mask_b).sum() / (skel_a.sum() + EPS)
    t_sens = (skel_b & mask_a).sum() / (skel_b.sum() + EPS)
    return float(2 * t_prec * t_sens / (t_prec + t_sens + EPS))


def compute_metrics(prob_raw: np.ndarray, prob_clahe: np.ndarray) -> dict:
    mask_raw = prob_raw >= PRED_THRESHOLD
    mask_clahe = prob_clahe >= PRED_THRESHOLD

    total_px = mask_raw.size
    fg_union = mask_raw | mask_clahe
    n_fg_union = int(fg_union.sum())

    hard_diff = mask_raw != mask_clahe

    D_all = float(hard_diff.sum() / total_px)

    if n_fg_union > 0:
        D_fg_union = float(hard_diff[fg_union].sum() / n_fg_union)
        D_prob_fg = float(np.abs(prob_raw[fg_union] - prob_clahe[fg_union]).mean())
    else:
        D_fg_union = 0.0
        D_prob_fg = 0.0

    skel_union = skeletonize(fg_union)
    n_skel = int(skel_union.sum())
    if n_skel > 0:
        D_skeleton = float(hard_diff[skel_union].sum() / n_skel)
    else:
        D_skeleton = 0.0

    cl_overlap = clskeleton_overlap(mask_raw, mask_clahe)

    n_raw = int(mask_raw.sum())
    n_clahe = int(mask_clahe.sum())
    fg_raw = n_raw / total_px
    fg_clahe = n_clahe / total_px
    fg_ratio_change = (n_clahe - n_raw) / (n_raw + EPS)

    return {
        "D_all": D_all,
        "D_fg_union": D_fg_union,
        "D_prob_fg": D_prob_fg,
        "D_skeleton": D_skeleton,
        "clskeleton_overlap": cl_overlap,
        "fg_raw": fg_raw,
        "fg_clahe": fg_clahe,
        "fg_ratio_change": fg_ratio_change,
        "n_fg_union_px": n_fg_union,
    }


# ============================================================
# Main
# ============================================================

def run_for_shift(domain_shift: str, device: str, out_dir: str, max_images):
    resolution = FROZEN_RESOLUTION[domain_shift]
    model = load_frozen_source_model(domain_shift, device=device)
    model.eval()

    image_paths = get_target_image_paths(domain_shift)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if len(image_paths) == 0:
        raise RuntimeError(f"Khong tim thay anh target nao cho {domain_shift}")

    rows = []
    for path in image_paths:
        img_bgr = load_image_bgr(path)
        img_clahe = apply_clahe_lab(img_bgr)

        prob_raw = predict_prob(model, img_bgr, resolution, device)
        prob_clahe = predict_prob(model, img_clahe, resolution, device)

        metrics = compute_metrics(prob_raw, prob_clahe)
        metrics["image"] = os.path.basename(path)
        metrics["domain_shift"] = domain_shift
        rows.append(metrics)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"raw_clahe_disagreement_{domain_shift}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{domain_shift}] {len(df)} anh -> {csv_path}")
    return df


def summarize(df: pd.DataFrame, domain_shift: str) -> dict:
    metric_cols = [
        "D_all", "D_fg_union", "D_prob_fg", "D_skeleton",
        "clskeleton_overlap", "fg_raw", "fg_clahe", "fg_ratio_change",
    ]
    summary = {"domain_shift": domain_shift, "n_images": len(df)}
    for col in metric_cols:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std()
        summary[f"{col}_median"] = df[col].median()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain_shifts", nargs="+",
        default=list(FROZEN_RESOLUTION.keys()),
        choices=list(FROZEN_RESOLUTION.keys()),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/raw_clahe_diagnostic")
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Gioi han so anh moi shift (de test code nhanh); mac dinh dung het target set.",
    )
    args = parser.parse_args()

    all_summaries = []
    for shift in args.domain_shifts:
        df = run_for_shift(shift, args.device, args.out_dir, args.max_images)
        all_summaries.append(summarize(df, shift))

    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(args.out_dir, "summary_all_shifts.csv")
    summary_df.to_csv(summary_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n" + "=" * 100)
    print("BANG TOM TAT — Raw vs CLAHE appearance-view disagreement")
    print("(source model dong bang, KHONG doc nhan target)")
    print("=" * 100)
    print(summary_df[[
        "domain_shift", "n_images",
        "D_fg_union_mean", "D_fg_union_std", "D_fg_union_median",
        "D_prob_fg_mean",
        "D_skeleton_mean",
        "clskeleton_overlap_mean",
        "fg_ratio_change_mean", "fg_ratio_change_median",
    ]].to_string(index=False))

    lo, hi = DUAL_EMA_DISAGREEMENT_REFERENCE
    print(f"\nDoi chieu tham khao: Dual-EMA disagreement da do truoc day ~{lo}-{hi}.")
    print(f"Toan bo bang chi tiet: {summary_path}")
    print("\nDoc ket qua theo 4 cau hoi (KHONG dung threshold cung):")
    print(f"  1) D_fg_union co lon ro ret hon {lo}-{hi} khong?")
    print("  2) clskeleton_overlap THAP dong nghia divergence nam o CAU TRUC")
    print("     vessel; clskeleton_overlap CAO du D_fg_union lon nghia la 2")
    print("     view chi lech nguong/do day, chua chac la diversity huu ich.")
    print("  3) fg_ratio_change duong manh & nhat quan ca 4 shift -> canh bao")
    print("     CLAHE co the dang tao foreground expansion (artefact), khong")
    print("     phai tin hieu domain that.")
    print("  4) Pattern (D_fg_union, clskeleton_overlap, fg_ratio_change) co")
    print("     KHAC NHAU giua 4 shift, hay lai collapse thanh 1 hanh vi?")
    print("\nLuu y: day moi la Tang 0 (frozen-model). Neu Tang 0 pass, buoc")
    print("tiep theo la Tang 2 (control training 3-5 epoch, theo doi D_i qua")
    print("tung epoch) TRUOC KHI code full Appearance-Diverse Dual Teacher —")
    print("Tang 0 khong chung minh divergence song sot qua training.")


if __name__ == "__main__":
    main()
