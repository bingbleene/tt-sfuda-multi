"""
tier3_complementarity_audit.py

MUC DICH (Tier 3 — DUY NHAT duoc mo GT, chi de audit gia thuyet)
------------------------------------------------------------------
Tier 2 da xong: raw/CLAHE tao divergence song sot qua Stage II adaptation
o ca 2 shift pilot, nhung theo 2 kieu khac han nhau va NGUOC voi thu hang
Tier 1 (H->C song sot manh nhat du la case Tier 1 danh gia xau nhat).
Dieu nay chung minh "divergence song sot" KHONG dong nghia "divergence
huu ich" — chi co the phan biet 2 kha nang bang cach xem GT:

    (a) 2 expert THAT SU sua loi cho nhau (complementary)
    (b) 2 expert chi bat dong on dinh vi 1 ben sai theo kieu ben vung
        (vd CLAHE tao artefact ma model hoc phai, khong phai recovery
        thong tin that)

RANH GIOI — DAY LA DIEM DUY NHAT TRONG TOAN BO TIER 0-3 DUOC MO GT:
    - Chi chay sau khi Tier 0/1/2 da khoa hoan toan (config, checkpoint,
      quyet dinh dung/khong dung CLAHE, gate criteria) — KHONG dung ket
      qua Tier 3 de quay lai sua bat ky thanh phan nao cua Tier 0-2.
    - GT o day la AUDIT, khong phai SELECTION — khong co dong code nao
      trong file nay dung GT de chon config, train, hay quyet dinh gi
      ngoai in ra so lieu de nguoi doc tu danh gia.
    - Tai su dung checkpoint epoch_0 va epoch_3 DA CO SAN tu Tier 2
      (tier2_survival.py) — KHONG train them gi trong file nay.

CO SO LOGIC — VI SAO KHONG THE CO "CA HAI SAI" TRONG VUNG BAT DONG THUAN:
    Voi GT nhi phan, neu raw_pred != clahe_pred (bat dong), 2 gia tri doi
    lap nhau (0 vs 1), GT chi co the khop dung DUNG 1 trong 2 — khong co
    kha nang "ca hai sai" o day. "Ca hai sai" CHI xay ra o vung 2 model
    DONG THUAN nhung dong thuan SAI (ca hai cung du doan sai 1 gia tri).
    Vi vay file nay tach ro 2 vung:
        - Vung BAT DONG (disagreement): chi co 2 loai — raw dung/clahe sai
          hoac clahe dung/raw sai — luon cong lai = 100%.
        - Vung DONG THUAN (agreement): co the dung chung hoac sai chung
          (joint_wrong) — day la vung dang chu y neu ty le cao, vi no cho
          thay 2 model "cung sai theo cach giong nhau", khong phai bo
          sung thong tin cho nhau.

METRIC CHINH:
    Dice_raw, Dice_clahe            — Dice tung expert rieng
    Dice_oracle_fusion              — Dice neu co 1 router hoan hao chon
                                       dung expert tai moi pixel bat dong
                                       (= GT tai vung bat dong, gia tri
                                       chung tai vung dong thuan) — TRAN
                                       LY THUYET, khong phai method that.
    complementarity_gain (G_comp)   — Dice_oracle_fusion - max(Dice_raw,
                                       Dice_clahe). G_comp ~ 0 => dung xay
                                       fusion. G_comp lon => co tran that
                                       de khai thac.
    pct_raw_correct_in_disagreement / pct_clahe_correct_in_disagreement
                                     — ty le trong vung bat dong (cong = 1)
    pct_joint_wrong_in_agreement    — ty le "ca hai sai giong nhau" trong
                                       vung dong thuan — canh bao neu cao

Tinh o CA epoch_0 VA epoch_3 (theo yeu cau) de xem complementarity ceiling
co song sot qua adaptation hay khong, khong chi diversity thuan tuy.

CHAY (trong TT_SFUDA_2D/, sau khi Tier 2 da chay xong va checkpoint con
trong tier2_ckpt_dir — thuong la /kaggle/working/outputs/tier2_survival):
    python run.py tier3_complementarity_audit.py \
        --domain_shifts chase2rite hrf2chase \
        --tier2_ckpt_dir /kaggle/working/outputs/tier2_survival
"""

import os
import argparse
from glob import glob

import cv2
import numpy as np
import pandas as pd
import yaml
import torch

import archs
from patch_inference import normalize_like_dataset

# ============================================================
# CONFIG
# ============================================================

PILOT_SHIFTS = {
    "chase2rite": 768,
    "hrf2chase": 384,
}
SEEDS = ["A", "B"]
EPOCHS_TO_AUDIT = [0, 3]
THRESHOLD = 0.5
EPS = 1e-7


# ============================================================
# Model / preprocess — dung nguyen ban da xac nhan (Tier 0/1/2)
# ============================================================

def load_config(domain_shift: str) -> dict:
    source, target = domain_shift.split("2")
    cfg_path = os.path.join("models", f"{source}_unet", f"config_{target}_dualema.yml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_arch(cfg: dict, device: str):
    return archs.__dict__[cfg["arch"]](
        cfg["num_classes"], cfg["input_channels"], cfg["deep_supervision"]
    ).to(device)


def load_checkpoint_into(model, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def apply_clahe_lab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_for_model(img_bgr: np.ndarray, resolution: int) -> torch.Tensor:
    img_resized = cv2.resize(img_bgr, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img_norm = normalize_like_dataset(img_resized)
    return torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)


@torch.no_grad()
def predict_prob(model, img_bgr: np.ndarray, resolution: int, device: str) -> np.ndarray:
    x = preprocess_for_model(img_bgr, resolution).to(device)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    prob = torch.sigmoid(out)
    return prob.squeeze().detach().cpu().numpy().astype(np.float32)


def get_test_image_mask_paths(target: str, cfg: dict):
    """DUY NHAT noi trong toan bo Tier 0-3 doc target mask — chi o day,
    chi sau khi moi quyet dinh khac da khoa."""
    img_dir = os.path.join("inputs", target, "test", "images")
    mask_dir = os.path.join("inputs", target, "test", "masks", "0")
    img_paths = sorted(glob(os.path.join(img_dir, "*" + cfg["img_ext"])))
    pairs = []
    for p in img_paths:
        iid = os.path.splitext(os.path.basename(p))[0]
        mpath = os.path.join(mask_dir, iid + cfg["mask_ext"])
        if os.path.exists(mpath):
            pairs.append((p, mpath))
    if not pairs:
        raise RuntimeError(f"Khong tim thay cap anh/mask nao trong {img_dir} / {mask_dir}")
    return pairs


def dice_score(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = int((pred_bool & gt_bool).sum())
    denom = int(pred_bool.sum()) + int(gt_bool.sum())
    return 1.0 if denom == 0 else 2.0 * inter / denom


# ============================================================
# Audit chinh — 1 cap (raw_model, clahe_model) tren toan bo test set
# ============================================================

def audit_pair(raw_model, clahe_model, resolution: int, device: str, img_mask_pairs):
    rows = []
    for img_path, mask_path in img_mask_pairs:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            raise RuntimeError(f"Khong doc duoc: {img_path} / {mask_path}")
        h, w = img.shape[:2]
        clh = apply_clahe_lab(img)

        p_raw = predict_prob(raw_model, img, resolution, device)
        p_clh = predict_prob(clahe_model, clh, resolution, device)
        p_raw = cv2.resize(p_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        p_clh = cv2.resize(p_clh, (w, h), interpolation=cv2.INTER_LINEAR)

        gt_bool = gt > 127
        mask_raw = p_raw >= THRESHOLD
        mask_clh = p_clh >= THRESHOLD

        raw_correct = mask_raw == gt_bool
        clh_correct = mask_clh == gt_bool

        disagree = mask_raw != mask_clh
        agree = ~disagree
        n_dis = int(disagree.sum())
        n_agree = int(agree.sum())

        # --- Vung BAT DONG: chi 2 loai, luon cong = 100% ---
        pct_raw_correct_dis = float(raw_correct[disagree].sum() / n_dis) if n_dis > 0 else float("nan")
        pct_clahe_correct_dis = float(clh_correct[disagree].sum() / n_dis) if n_dis > 0 else float("nan")

        # --- Vung DONG THUAN: co the dung chung hoac sai chung ---
        joint_correct = raw_correct & agree   # (== clh_correct & agree, vi mask_raw==mask_clh o day)
        joint_wrong = (~raw_correct) & agree
        pct_joint_correct_agree = float(joint_correct.sum() / n_agree) if n_agree > 0 else float("nan")
        pct_joint_wrong_agree = float(joint_wrong.sum() / n_agree) if n_agree > 0 else float("nan")

        dice_raw = dice_score(mask_raw, gt_bool)
        dice_clahe = dice_score(mask_clh, gt_bool)

        # Oracle pixel-fusion: tai vung bat dong dung GT (luon dung, vi
        # dung 1 trong 2 expert da dung); tai vung dong thuan giu gia tri
        # chung (khong co gi de chon, ca hai giong nhau).
        oracle_pred = np.where(disagree, gt_bool, mask_raw)
        dice_oracle = dice_score(oracle_pred.astype(bool), gt_bool)

        g_comp = dice_oracle - max(dice_raw, dice_clahe)

        rows.append({
            "image": os.path.basename(img_path),
            "n_disagree_px": n_dis,
            "n_agree_px": n_agree,
            "pct_raw_correct_in_disagreement": pct_raw_correct_dis,
            "pct_clahe_correct_in_disagreement": pct_clahe_correct_dis,
            "pct_joint_correct_in_agreement": pct_joint_correct_agree,
            "pct_joint_wrong_in_agreement": pct_joint_wrong_agree,
            "dice_raw": dice_raw,
            "dice_clahe": dice_clahe,
            "dice_oracle_fusion": dice_oracle,
            "complementarity_gain": g_comp,
        })
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def run_shift(domain_shift: str, resolution: int, device: str, tier2_ckpt_dir: str, out_dir: str):
    cfg = load_config(domain_shift)
    _, target = domain_shift.split("2")
    img_mask_pairs = get_test_image_mask_paths(target, cfg)
    print(f"[{domain_shift}] {len(img_mask_pairs)} cap anh/mask (test set)")

    all_rows = []
    for seed in SEEDS:
        for epoch in EPOCHS_TO_AUDIT:
            raw_ckpt = os.path.join(tier2_ckpt_dir, domain_shift, f"raw_{seed}", f"epoch_{epoch}.pth")
            clahe_ckpt = os.path.join(tier2_ckpt_dir, domain_shift, f"clahe_{seed}", f"epoch_{epoch}.pth")
            for path in (raw_ckpt, clahe_ckpt):
                if not os.path.exists(path):
                    raise RuntimeError(f"Thieu checkpoint Tier 2: {path} (chay tier2_survival.py truoc)")

            raw_model = load_arch(cfg, device)
            load_checkpoint_into(raw_model, raw_ckpt, device)
            clahe_model = load_arch(cfg, device)
            load_checkpoint_into(clahe_model, clahe_ckpt, device)

            df = audit_pair(raw_model, clahe_model, resolution, device, img_mask_pairs)
            df["domain_shift"] = domain_shift
            df["seed"] = seed
            df["epoch"] = epoch
            all_rows.append(df)
            print(f"[{domain_shift} seed={seed} epoch={epoch}] "
                  f"dice_raw={df['dice_raw'].mean():.4f} dice_clahe={df['dice_clahe'].mean():.4f} "
                  f"dice_oracle={df['dice_oracle_fusion'].mean():.4f} "
                  f"G_comp={df['complementarity_gain'].mean():.4f}")

    full_df = pd.concat(all_rows, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"tier3_audit_{domain_shift}.csv")
    full_df.to_csv(csv_path, index=False)
    return full_df


def summarize(full_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "pct_raw_correct_in_disagreement", "pct_clahe_correct_in_disagreement",
        "pct_joint_correct_in_agreement", "pct_joint_wrong_in_agreement",
        "dice_raw", "dice_clahe", "dice_oracle_fusion", "complementarity_gain",
    ]
    summary = full_df.groupby(["domain_shift", "seed", "epoch"])[metric_cols].mean().reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_shifts", nargs="+", default=list(PILOT_SHIFTS.keys()),
                         choices=list(PILOT_SHIFTS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tier2_ckpt_dir", default="/kaggle/working/outputs/tier2_survival")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/tier3_audit")
    args = parser.parse_args()

    all_full = []
    for shift in args.domain_shifts:
        resolution = PILOT_SHIFTS[shift]
        df = run_shift(shift, resolution, args.device, args.tier2_ckpt_dir, args.out_dir)
        all_full.append(df)

    full_df = pd.concat(all_full, ignore_index=True)
    summary = summarize(full_df)
    summary_path = os.path.join(args.out_dir, "tier3_summary.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 110)
    print("TIER 3 COMPLEMENTARITY AUDIT (GT mo CHI de audit, khong feed vao algorithm)")
    print("=" * 110)
    print(summary.to_string(index=False))
    print(f"\nChi tiet per-image: {args.out_dir}/tier3_audit_<shift>.csv")
    print(f"Bang tom tat: {summary_path}")

    print("\nDoc ket qua:")
    print("  - pct_raw_correct_in_disagreement + pct_clahe_correct_in_disagreement = 100%")
    print("    (khong the co 'ca hai sai' trong vung bat dong voi GT nhi phan)")
    print("  - pct_joint_wrong_in_agreement CAO => 2 model dong thuan SAI cung nhau")
    print("    o nhieu pixel — dau hieu chia se cung 1 loai loi, khong bu tru")
    print("  - complementarity_gain (G_comp) ~ 0 tai epoch 3 => DUNG xay fusion,")
    print("    du Tier 2 cho divergence song sot cao (dung la truong hop H->C)")
    print("  - G_comp lon (vai diem Dice) va GIU NGUYEN hoac tang tu epoch 0 den")
    print("    epoch 3 => co tran that de khai thac, dang xay AppearanceDiverseTeacher")
    print("  - So sanh G_comp(epoch 0) vs G_comp(epoch 3): neu giam manh, complementarity")
    print("    dang 'mon di' qua adaptation dung nhu diversity co the mon di (xem Tier 2)")


if __name__ == "__main__":
    main()
