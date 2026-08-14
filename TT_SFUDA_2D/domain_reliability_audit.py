"""
domain_reliability_audit.py

MUC DICH — kiem tra: flip-reliability o muc DOMAIN (trung binh tren target
TRAIN images khong nhan) co du doan duoc expert nao (raw/CLAHE) manh hon
o TEST set khong. KHONG train gi, KHONG combine/tune/threshold — chi bao
dau (sign) va do lon cua ΔR, doi chieu voi Dice that.

DAY LA CAU HOI DAU TIEN TRUOC KHI XAY GATE:
    "Flip consistency co du doan duoc expert dominance khong?"
Neu KHONG, dung ep no lam 3-way gate (raw/fusion/clahe).

RANH GIOI:
    - ΔR (R_all, R_fg) tinh HOAN TOAN tu target/train/images (khong nhan)
      — day la phan "unsupervised signal", co the dung THAT trong deploy.
    - Dice_raw - Dice_clahe tinh tu target/test (co GT) — CHI DE AUDIT
      xem dau cua ΔR co dung khong, KHONG dung de chon/sua ΔR hay bat ky
      thanh phan nao khac.
    - KHONG combine R_all/R_fg thanh 1 so, KHONG threshold, KHONG weight
      — bao ca hai rieng biet, de nguoi doc tu danh gia.
    - Luu ΔR TUNG ANH (khong chi trung binh domain) vao CSV rieng — de
      buoc sau (CI-bootstrap, neu audit nay pass) khong can suy luan lai
      model.

CHAY (trong TT_SFUDA_2D/, sau khi da co 24 checkpoint full-budget):
    python run.py domain_reliability_audit.py \
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

ALL_SHIFTS = {
    "chase2hrf": 1024,
    "chase2rite": 768,
    "hrf2chase": 384,
    "hrf2rite": 512,
}
THRESHOLD = 0.5
EPS = 1e-7


def load_config(domain_shift: str) -> dict:
    source, target = domain_shift.split("2")
    cfg_path = os.path.join("models", f"{source}_unet", f"config_{target}_dualema.yml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
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


def build_transform(cfg: dict, use_clahe: bool) -> Compose:
    """Y HET val_transform that — deterministic, KHONG RandomRotate90/Flip
    (do la augmentation ngau nhien chi dung cho train_transform, khong
    phu hop tinh mot tin hieu on dinh cho tung anh)."""
    return Compose([
        *([ClaheLAB(clip_limit=2.0)] if use_clahe else []),
        Resize(cfg["input_h"], cfg["input_w"]),
        transforms.Normalize(),
    ])


def get_loader(cfg: dict, target: str, split: str, use_clahe: bool):
    img_dir = os.path.join("inputs", target, split, "images")
    mask_dir = os.path.join("inputs", target, split, "masks")
    img_ids = sorted(glob(os.path.join(img_dir, "*" + cfg["img_ext"])))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    transform = build_transform(cfg, use_clahe)
    dataset = Dataset(
        img_ids=img_ids, img_dir=img_dir, mask_dir=mask_dir,
        img_ext=cfg["img_ext"], mask_ext=cfg["mask_ext"],
        num_classes=cfg["num_classes"], transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=cfg["num_workers"], drop_last=False,
    )
    return loader, img_ids


def dice_score(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = int((pred_bool & gt_bool).sum())
    denom = int(pred_bool.sum()) + int(gt_bool.sum())
    return 1.0 if denom == 0 else 2.0 * inter / denom


@torch.no_grad()
def model_forward(model, inp, device):
    inp = inp.to(device)
    out = model(inp)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return torch.sigmoid(out).squeeze().cpu().numpy()


@torch.no_grad()
def predict_with_flip(model, inp: torch.Tensor, device: str):
    p = model_forward(model, inp, device)
    inp_flip = torch.flip(inp, dims=[-1])
    p_flip = model_forward(model, inp_flip, device)
    p_flip = np.flip(p_flip, axis=-1).copy()
    return p, p_flip


# ============================================================
# (1) Unsupervised signal — DeltaR tren TRAIN images (khong nhan)
# ============================================================

def compute_domain_reliability(raw_model, clahe_model, cfg, target, device):
    raw_loader, img_ids = get_loader(cfg, target, "train", use_clahe=False)
    clahe_loader, img_ids_c = get_loader(cfg, target, "train", use_clahe=True)
    assert img_ids == img_ids_c, "Thu tu anh raw/clahe train loader khong khop."

    rows = []
    for idx, (batch_raw, batch_clh) in enumerate(zip(raw_loader, clahe_loader)):
        inp_raw = batch_raw[0]
        inp_clh = batch_clh[0]

        p_raw, p_raw_flip = predict_with_flip(raw_model, inp_raw, device)
        p_clh, p_clh_flip = predict_with_flip(clahe_model, inp_clh, device)

        r_raw_px = 1.0 - np.abs(p_raw - p_raw_flip)
        r_clh_px = 1.0 - np.abs(p_clh - p_clh_flip)

        r_raw_all = float(r_raw_px.mean())
        r_clh_all = float(r_clh_px.mean())

        fg_union = (p_raw >= THRESHOLD) | (p_clh >= THRESHOLD)
        if fg_union.sum() > 0:
            r_raw_fg = float(r_raw_px[fg_union].mean())
            r_clh_fg = float(r_clh_px[fg_union].mean())
        else:
            r_raw_fg = float("nan")
            r_clh_fg = float("nan")

        rows.append({
            "image": img_ids[idx],
            "R_raw_all": r_raw_all, "R_clahe_all": r_clh_all,
            "delta_R_all": r_raw_all - r_clh_all,
            "R_raw_fg": r_raw_fg, "R_clahe_fg": r_clh_fg,
            "delta_R_fg": r_raw_fg - r_clh_fg,
        })
    return pd.DataFrame(rows)


# ============================================================
# (2) Audit reference — Dice_raw, Dice_clahe tren TEST (co GT)
#     CHI DE DOI CHIEU DAU, khong feed nguoc vao (1)
# ============================================================

def compute_test_dice(raw_model, clahe_model, cfg, target, device):
    raw_loader, img_ids = get_loader(cfg, target, "test", use_clahe=False)
    clahe_loader, img_ids_c = get_loader(cfg, target, "test", use_clahe=True)
    assert img_ids == img_ids_c

    dice_raw_list, dice_clahe_list = [], []
    for batch_raw, batch_clh in zip(raw_loader, clahe_loader):
        inp_raw, tgt_raw = batch_raw[0], batch_raw[1]
        inp_clh = batch_clh[0]
        gt_bool = (tgt_raw.numpy().squeeze() >= 0.5)

        p_raw = model_forward(raw_model, inp_raw, device)
        p_clh = model_forward(clahe_model, inp_clh, device)

        dice_raw_list.append(dice_score(p_raw >= THRESHOLD, gt_bool))
        dice_clahe_list.append(dice_score(p_clh >= THRESHOLD, gt_bool))

    return float(np.mean(dice_raw_list)), float(np.mean(dice_clahe_list))


def run_shift_seed(domain_shift, seed, device, ckpt_root, out_dir):
    cfg = load_config(domain_shift)
    _, target = domain_shift.split("2")

    raw_ckpt = os.path.join(ckpt_root, domain_shift, f"raw_seed{seed}", "model.pth")
    clahe_ckpt = os.path.join(ckpt_root, domain_shift, f"clahe_seed{seed}", "model.pth")
    for p in (raw_ckpt, clahe_ckpt):
        if not os.path.exists(p):
            raise RuntimeError(f"Thieu checkpoint: {p}")

    raw_model = load_arch(cfg, device)
    load_checkpoint_into(raw_model, raw_ckpt, device)
    clahe_model = load_arch(cfg, device)
    load_checkpoint_into(clahe_model, clahe_ckpt, device)

    per_image_df = compute_domain_reliability(raw_model, clahe_model, cfg, target, device)
    per_image_df["domain_shift"] = domain_shift
    per_image_df["seed"] = seed

    os.makedirs(out_dir, exist_ok=True)
    per_image_df.to_csv(
        os.path.join(out_dir, f"domain_reliability_perimg_{domain_shift}_seed{seed}.csv"),
        index=False,
    )

    dice_raw, dice_clahe = compute_test_dice(raw_model, clahe_model, cfg, target, device)

    summary_row = {
        "domain_shift": domain_shift, "seed": seed,
        "delta_R_all_mean": per_image_df["delta_R_all"].mean(),
        "delta_R_fg_mean": per_image_df["delta_R_fg"].mean(),
        "delta_R_all_std": per_image_df["delta_R_all"].std(),
        "delta_R_fg_std": per_image_df["delta_R_fg"].std(),
        "n_train_images": len(per_image_df),
        "dice_raw_test": dice_raw,
        "dice_clahe_test": dice_clahe,
        "dice_diff_raw_minus_clahe": dice_raw - dice_clahe,
    }
    return summary_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_shifts", nargs="+", default=list(ALL_SHIFTS.keys()),
                         choices=list(ALL_SHIFTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_root", default="/kaggle/working/outputs/full_checkpoints")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/domain_reliability")
    args = parser.parse_args()

    summary_rows = []
    for shift in args.domain_shifts:
        for seed in args.seeds:
            print(f"[{shift} seed={seed}] dang audit domain reliability...")
            row = run_shift_seed(shift, seed, args.device, args.ckpt_root, args.out_dir)
            summary_rows.append(row)
            print(f"  DeltaR_all={row['delta_R_all_mean']:.4f}  DeltaR_fg={row['delta_R_fg_mean']:.4f}  "
                  f"Dice_raw-Dice_clahe={row['dice_diff_raw_minus_clahe']:.4f}")

    summary = pd.DataFrame(summary_rows)
    summary["sign_match_all"] = (
        np.sign(summary["delta_R_all_mean"]) == np.sign(summary["dice_diff_raw_minus_clahe"])
    )
    summary["sign_match_fg"] = (
        np.sign(summary["delta_R_fg_mean"]) == np.sign(summary["dice_diff_raw_minus_clahe"])
    )

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "domain_reliability_summary.csv")
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 120)
    print("DOMAIN RELIABILITY AUDIT — DeltaR (train, khong nhan) vs Dice_raw-Dice_clahe (test, GT audit)")
    print("=" * 120)
    print(summary[["domain_shift", "seed", "delta_R_all_mean", "delta_R_fg_mean",
                    "dice_diff_raw_minus_clahe", "sign_match_all", "sign_match_fg"]].to_string(index=False))

    n_total = len(summary)
    n_match_all = int(summary["sign_match_all"].sum())
    n_match_fg = int(summary["sign_match_fg"].sum())
    print(f"\nSign match DeltaR_all: {n_match_all}/{n_total}")
    print(f"Sign match DeltaR_fg:  {n_match_fg}/{n_total}")
    print(f"\nChi tiet: {summary_path}")
    print("Per-image DeltaR (de dung cho CI-bootstrap sau, khong can suy luan lai):")
    print(f"  {args.out_dir}/domain_reliability_perimg_<shift>_seed<n>.csv")

    print("\nDoc ket qua:")
    print("  - Neu sign_match cao va NHAT QUAN qua ca 3 seed cho MOI shift")
    print("    -> flip-reliability domain-level co du doan duoc dominance,")
    print("    dang thu buoc CI-based 3-way gate tiep theo.")
    print("  - Neu sign_match thap hoac khong nhat quan giua cac seed cua")
    print("    CUNG 1 shift -> tin hieu khong du on dinh, KHONG nen ep")
    print("    thanh gate — day van la ket luan khoa hoc sach.")


if __name__ == "__main__":
    main()
