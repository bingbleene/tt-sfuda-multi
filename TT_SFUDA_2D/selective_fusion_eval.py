"""
selective_fusion_eval.py  (v2 — model-resolution evaluation, sua bug native-upsample)

MUC DICH — danh gia Flip-Reliability Selective Fusion (MAD rule) tren
checkpoint full-budget da train doc lap hoan toan (Phuong an A), KHONG
train them gi trong file nay.

SUA SO VOI BAN CU (v1): v1 du doan o model resolution roi UPSAMPLE prediction
len NATIVE resolution truoc khi so voi native GT — KHAC voi validate() that
cua repo (danh gia hoan toan o model resolution). Da xac nhan bang
eval_protocol_audit.py: HRF->CHASE raw seed1 A(repo)=0.6575, C(v1,
native-upsample)=0.6270 — chenh +0.0305, THUAN TUY do evaluation-space
mismatch, KHONG phai training regression.

v2 nay: raw prediction, CLAHE prediction, R_flip, mean, router, oracle —
TAT CA deu tinh o dung MODEL RESOLUTION da freeze cho tung shift. GT lay
qua chinh Dataset/Compose(Resize+Normalize) cua repo — khong doc anh/mask
bang cv2.imread rieng, khong co buoc resize/upsample nao khac ngoai chinh
pipeline training da dung.

RANH GIOI (giu nguyen tu v1):
    - GT chi dung de tinh Dice cuoi cung va oracle ceiling (audit) — KHONG
      dung de chon MAD coefficient/tau. Cong thuc MAD la DUY NHAT, khoa
      cung, khong co he so nao de tune.
    - Checkpoint phai la FULL-BUDGET (15 epoch + early_stop_unsupervised
      that).

ROUTER — MAD rule (khoa cung, khong tune, khong doi so voi v1):
    Tren VUNG BAT DONG cua TUNG ANH:
        g_i = |R_r(i) - R_c(i)|          R = flip-TTA reliability = 1-U_tta
        tau_img = median(g) + MAD(g)     MAD(g) = median(|g - median(g)|)
    R_r(i)-R_c(i) > tau_img  -> p_fused(i) = p_raw(i)
    R_c(i)-R_r(i) > tau_img  -> p_fused(i) = p_clahe(i)
    Nguoc lai (abstain, gom vung dong thuan) -> p_fused(i) = (p_raw+p_clahe)/2

4 METHOD SO SANH: Raw branch, CLAHE branch, Mean Ensemble, Selective Fusion (MAD)
    + oracle ceiling (audit only)

BANG "PAPER COMPARISON" TU KET QUA FILE NAY: moi shift duoc adapt VA danh
gia o dung resolution da chon boi CMR (source-free spatial calibration) —
KHONG phai tat ca o 512. Khi viet luan van, noi ro dieu nay (xem thao luan
da thong nhat), khong ngam dinh "controlled @512".

CHAY (trong TT_SFUDA_2D/, sau khi da co checkpoint full-budget):
    python run.py selective_fusion_eval.py \
        --domain_shifts chase2hrf chase2rite hrf2chase hrf2rite \
        --seeds 1 2 3 \
        --ckpt_root /kaggle/working/outputs/full_checkpoints
"""

import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import yaml
import torch

import archs
from dataset import Dataset
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from clahe_transform import ClaheLAB

# ============================================================
# CONFIG
# ============================================================

ALL_SHIFTS = {
    "chase2hrf": 1024,
    "chase2rite": 768,
    "hrf2chase": 384,
    "hrf2rite": 512,
}
THRESHOLD = 0.5
EPS = 1e-7


# ============================================================
# Config / Dataset — dung nguyen ban repo, khong viet lai
# ============================================================

def load_config(domain_shift: str) -> dict:
    source, target = domain_shift.split("2")
    cfg_path = os.path.join("models", f"{source}_unet", f"config_{target}_dualema.yml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # QUAN TRONG: ghi de input_h/input_w bang resolution DA FREEZE cho
    # shift nay — giong het --input_size lam luc train. Neu khong lam
    # buoc nay se lap lai dung bug da phat hien o eval_protocol_audit.py
    # (danh gia nham o resolution mac dinh 512 trong YAML goc).
    resolution = ALL_SHIFTS[domain_shift]
    cfg["input_h"] = resolution
    cfg["input_w"] = resolution
    return cfg


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


def build_val_transform(cfg: dict, use_clahe: bool) -> Compose:
    """Y HET val_transform trong tt_sfuda_2d_dualema.py — khong viet lai khac."""
    return Compose([
        *([ClaheLAB(clip_limit=2.0)] if use_clahe else []),
        Resize(cfg["input_h"], cfg["input_w"]),
        transforms.Normalize(),
    ])


def get_val_loader(cfg: dict, target: str, use_clahe: bool):
    val_img_ids = sorted(glob(os.path.join("inputs", target, "test", "images", "*" + cfg["img_ext"])))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    val_transform = build_val_transform(cfg, use_clahe)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join("inputs", target, "test", "images"),
        mask_dir=os.path.join("inputs", target, "test", "masks"),
        img_ext=cfg["img_ext"], mask_ext=cfg["mask_ext"],
        num_classes=cfg["num_classes"], transform=val_transform,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=cfg["num_workers"], drop_last=False,
    )
    return val_loader, val_img_ids


def dice_score(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = int((pred_bool & gt_bool).sum())
    denom = int(pred_bool.sum()) + int(gt_bool.sum())
    return 1.0 if denom == 0 else 2.0 * inter / denom


@torch.no_grad()
def predict_with_flip(model, inp: torch.Tensor, device: str):
    """inp: tensor da qua transform (model resolution, normalized).
    Tra ve (p, p_flip) numpy 2D, cung kich thuoc model resolution."""
    inp = inp.to(device)
    out = model(inp)
    if isinstance(out, (tuple, list)):
        out = out[0]
    p = torch.sigmoid(out).squeeze().cpu().numpy()

    inp_flip = torch.flip(inp, dims=[-1])
    out_flip = model(inp_flip)
    if isinstance(out_flip, (tuple, list)):
        out_flip = out_flip[0]
    p_flip = torch.sigmoid(out_flip).squeeze().cpu().numpy()
    p_flip = np.flip(p_flip, axis=-1).copy()  # lat lai ve dung chieu goc

    return p, p_flip


# ============================================================
# MAD selective router — khoa cung, khong doi so voi v1
# ============================================================

def mad_selective_fuse(p_raw, p_clh, r_raw, r_clh, disagree):
    p_fused = (p_raw + p_clh) / 2.0
    if disagree.sum() == 0:
        return p_fused, float("nan"), 0, 0

    g = np.abs(r_raw - r_clh)[disagree]
    med_g = np.median(g)
    mad_g = np.median(np.abs(g - med_g))
    tau_img = med_g + mad_g

    gap = r_raw - r_clh
    route_raw = disagree & (gap > tau_img)
    route_clh = disagree & (-gap > tau_img)

    p_fused = np.where(route_raw, p_raw, p_fused)
    p_fused = np.where(route_clh, p_clh, p_fused)
    return p_fused, tau_img, int(route_raw.sum()), int(route_clh.sum())


# ============================================================
# Danh gia 1 shift, 1 seed — TAT CA o model resolution
# ============================================================

def evaluate_pair(raw_model, clahe_model, cfg, target, device):
    raw_loader, val_img_ids = get_val_loader(cfg, target, use_clahe=False)
    clahe_loader, val_img_ids_c = get_val_loader(cfg, target, use_clahe=True)
    assert val_img_ids == val_img_ids_c, "Thu tu anh raw/clahe loader khong khop nhau."

    rows = []
    for idx, (batch_raw, batch_clh) in enumerate(zip(raw_loader, clahe_loader)):
        inp_raw, tgt_raw = batch_raw[0], batch_raw[1]
        inp_clh, tgt_clh = batch_clh[0], batch_clh[1]

        gt_raw_np = (tgt_raw.numpy().squeeze() >= 0.5)
        gt_clh_np = (tgt_clh.numpy().squeeze() >= 0.5)
        assert np.array_equal(gt_raw_np, gt_clh_np), (
            f"GT lech giua raw/clahe loader tai anh {val_img_ids[idx]} — "
            f"kiem tra lai ClaheLAB co vo tinh dung vao mask khong."
        )
        gt_bool = gt_raw_np

        p_raw, p_raw_flip = predict_with_flip(raw_model, inp_raw, device)
        p_clh, p_clh_flip = predict_with_flip(clahe_model, inp_clh, device)

        r_raw = 1.0 - np.abs(p_raw - p_raw_flip)
        r_clh = 1.0 - np.abs(p_clh - p_clh_flip)

        mask_raw = p_raw >= THRESHOLD
        mask_clh = p_clh >= THRESHOLD
        disagree = mask_raw != mask_clh

        p_mean = (p_raw + p_clh) / 2.0
        p_router, tau_img, n_route_raw, n_route_clh = mad_selective_fuse(
            p_raw, p_clh, r_raw, r_clh, disagree)

        dice_raw = dice_score(mask_raw, gt_bool)
        dice_clahe = dice_score(mask_clh, gt_bool)
        dice_mean = dice_score(p_mean >= THRESHOLD, gt_bool)
        dice_router = dice_score(p_router >= THRESHOLD, gt_bool)

        oracle_pred = np.where(disagree, gt_bool, mask_raw)
        dice_oracle = dice_score(oracle_pred.astype(bool), gt_bool)

        rows.append({
            "image": val_img_ids[idx],
            "n_disagree_px": int(disagree.sum()),
            "tau_img": tau_img,
            "n_routed_raw": n_route_raw,
            "n_routed_clahe": n_route_clh,
            "dice_raw": dice_raw,
            "dice_clahe": dice_clahe,
            "dice_mean": dice_mean,
            "dice_router": dice_router,
            "dice_oracle": dice_oracle,
        })
    return pd.DataFrame(rows)


def run_shift_seed(domain_shift, seed, device, ckpt_root, out_dir):
    cfg = load_config(domain_shift)
    _, target = domain_shift.split("2")

    raw_ckpt = os.path.join(ckpt_root, domain_shift, f"raw_seed{seed}", "model.pth")
    clahe_ckpt = os.path.join(ckpt_root, domain_shift, f"clahe_seed{seed}", "model.pth")
    for p in (raw_ckpt, clahe_ckpt):
        if not os.path.exists(p):
            raise RuntimeError(f"Thieu checkpoint full-budget: {p}")

    raw_model = load_arch(cfg, device)
    load_checkpoint_into(raw_model, raw_ckpt, device)
    clahe_model = load_arch(cfg, device)
    load_checkpoint_into(clahe_model, clahe_ckpt, device)

    df = evaluate_pair(raw_model, clahe_model, cfg, target, device)
    df["domain_shift"] = domain_shift
    df["seed"] = seed
    df["eval_resolution"] = cfg["input_h"]

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"selective_fusion_{domain_shift}_seed{seed}.csv"), index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_shifts", nargs="+", default=list(ALL_SHIFTS.keys()),
                         choices=list(ALL_SHIFTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_root", default="/kaggle/working/outputs/full_checkpoints")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/selective_fusion")
    args = parser.parse_args()

    all_dfs = []
    for shift in args.domain_shifts:
        for seed in args.seeds:
            print(f"[{shift} seed={seed}] danh gia o resolution={ALL_SHIFTS[shift]}...")
            df = run_shift_seed(shift, seed, args.device, args.ckpt_root, args.out_dir)
            all_dfs.append(df)
            print(f"[{shift} seed={seed}] dice_raw={df.dice_raw.mean():.4f} "
                  f"dice_clahe={df.dice_clahe.mean():.4f} dice_mean={df.dice_mean.mean():.4f} "
                  f"dice_router={df.dice_router.mean():.4f}")

    full_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    full_path = os.path.join(args.out_dir, "selective_fusion_all.csv")
    full_df.to_csv(full_path, index=False)

    summary = full_df.groupby(["domain_shift", "seed"]).agg(
        eval_resolution=("eval_resolution", "first"),
        dice_raw=("dice_raw", "mean"),
        dice_clahe=("dice_clahe", "mean"),
        dice_mean=("dice_mean", "mean"),
        dice_router=("dice_router", "mean"),
        dice_oracle=("dice_oracle", "mean"),
    ).reset_index()

    summary["delta_router_vs_mean"] = summary["dice_router"] - summary["dice_mean"]
    summary["best_single"] = summary[["dice_raw", "dice_clahe"]].max(axis=1)
    summary["G_oracle"] = summary["dice_oracle"] - summary["best_single"]
    summary["G_real"] = summary["dice_router"] - summary["best_single"]
    summary["ceiling_utilization_U"] = summary["G_real"] / (summary["G_oracle"] + EPS)

    summary_path = os.path.join(args.out_dir, "selective_fusion_summary.csv")
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 130)
    print("SELECTIVE FUSION v2 (model-resolution evaluation) — 4 METHOD + CEILING UTILIZATION")
    print("=" * 130)
    print(summary.to_string(index=False))

    print("\nDoc ket qua:")
    print("  - dice_router phai > dice_mean o phan lon shift/seed -> gia tri rieng cua router")
    print("  - so voi paper: dung dung dice_raw/dice_clahe/dice_router O DAY, KHONG dung so lieu")
    print("    tu selective_fusion_eval.py v1 (native-upsample, da xac nhan sai lech)")
    print(f"\nChi tiet per-image: {args.out_dir}/selective_fusion_<shift>_seed<n>.csv")
    print(f"Bang tong hop: {summary_path}")


if __name__ == "__main__":
    main()
