"""
fusion_benefit_audit.py

MUC DICH — TERMINAL HYPOTHESIS TEST cua development phase. Day la audit
RE cuoi cung truoc khi dong huong "automatic gate" (neu fail) hoac ghi
nhan "development observation" (neu dep, van KHONG dung lam gate that
neu chua kiem chung tren domain shift moi).

GIA THUYET (phat bieu TRUOC khi nhin B):
    directional symmetry of disagreement -> potential benefit from fusion

B KHONG PHAI RELIABILITY — no la DIRECTIONAL SYMMETRY:
    Trong vung bat dong, chi co 2 loai pixel:
        N10 = so pixel raw=1, clahe=0
        N01 = so pixel raw=0, clahe=1
        N_dis = N10 + N01
    B = 1 - |N10 - N01| / (N_dis + eps)
      ~ 2*min(N10,N01) / (N10+N01)
    B~0: bat dong gan nhu MOT CHIEU (vd CLAHE luon THEM foreground)
    B~1: bat dong CAN BANG HAI CHIEU
    B KHONG noi ben nao dung — chi noi bat dong co "hai huong" hay khong.

LY DO CO THE FAIL (da du bao TRUOC khi chay, khong phai bien minh sau):
    Tier 0 da cho thay CLAHE lam foreground tang manh o CA 4 SHIFT. Neu
    dieu do dung o moi domain (khong rieng C->R), B se chi tai phat hien
    "CLAHE co xu huong mo rong foreground" — KHONG phai tin hieu
    complementarity dac thu C->R. Day la falsification value that.

RANH GIOI KHOA CUNG (khong doi sau khi chay):
    - B tinh HOAN TOAN tu target TRAIN images (khong nhan).
    - GT (Y_s = mean thang best-single, cap SHIFT, da biet tu truoc) CHI
      in ra de doi chieu — KHONG dung trong bat ky phep tinh B nao.
    - KHONG threshold B de phan loai Yes/No — chi n=4 shift, khong du co
      so hoc threshold.
    - KHONG tao B2, B+entropy, B+Frangi, weighted-B, hay bat ky bien the
      nao sau khi xem ket qua. Neu fail, DUNG huong automatic gate.
    - Aggregation CHINH la macro (trung binh B_i tung anh). Pooled B chi
      la secondary diagnostic, KHONG duoc dung thay macro neu macro xau.
    - Anh khong co bat dong: B_i = NaN (khong phai 0 hay 1).
    - 3 seed trong 1 shift KHONG phai 3 datapoint doc lap — chi la do
      on dinh cua estimate B trong CUNG 1 shift (n=4 shift, khong phai
      n=12).

CHAY (trong TT_SFUDA_2D/, sau khi da co 24 checkpoint full-budget):
    python run.py fusion_benefit_audit.py \
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

# Tham khao CHI DE IN RA doi chieu cuoi cung — KHONG dung trong tinh B.
# Y_s = 1 neu Dice_mean > Dice_best_single (tu selective_fusion_v2, da co).
KNOWN_FUSION_BENEFIT_Y = {
    "chase2hrf": 0,
    "chase2rite": 1,
    "hrf2chase": 0,
    "hrf2rite": 0,
}


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


@torch.no_grad()
def model_forward(model, inp, device):
    inp = inp.to(device)
    out = model(inp)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return torch.sigmoid(out).squeeze().cpu().numpy()


# ============================================================
# B — directional disagreement symmetry, tren TRAIN images
# ============================================================

def compute_B_per_image(raw_model, clahe_model, cfg, target, device):
    raw_loader, img_ids = get_loader(cfg, target, "train", use_clahe=False)
    clahe_loader, img_ids_c = get_loader(cfg, target, "train", use_clahe=True)
    assert img_ids == img_ids_c, "Thu tu anh raw/clahe train loader khong khop."

    rows = []
    total_N10, total_N01, total_Ndis, total_px = 0, 0, 0, 0

    for idx, (batch_raw, batch_clh) in enumerate(zip(raw_loader, clahe_loader)):
        inp_raw = batch_raw[0]
        inp_clh = batch_clh[0]

        p_raw = model_forward(raw_model, inp_raw, device)
        p_clh = model_forward(clahe_model, inp_clh, device)

        mask_raw = p_raw >= THRESHOLD
        mask_clh = p_clh >= THRESHOLD

        n10 = int((mask_raw & ~mask_clh).sum())
        n01 = int((~mask_raw & mask_clh).sum())
        n_dis = n10 + n01
        n_px = mask_raw.size

        b_i = (1.0 - abs(n10 - n01) / (n_dis + EPS)) if n_dis > 0 else float("nan")
        c_d = n_dis / n_px

        total_N10 += n10
        total_N01 += n01
        total_Ndis += n_dis
        total_px += n_px

        rows.append({
            "image": img_ids[idx],
            "N10": n10, "N01": n01, "N_disagree": n_dis,
            "B_i": b_i, "coverage_C_D": c_d,
        })

    df = pd.DataFrame(rows)
    b_pooled = (1.0 - abs(total_N10 - total_N01) / (total_Ndis + EPS)) if total_Ndis > 0 else float("nan")
    return df, b_pooled, total_N10, total_N01, total_Ndis, total_px


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

    per_img_df, b_pooled, N10, N01, Ndis, Npx = compute_B_per_image(
        raw_model, clahe_model, cfg, target, device)
    per_img_df["domain_shift"] = domain_shift
    per_img_df["seed"] = seed

    os.makedirs(out_dir, exist_ok=True)
    per_img_df.to_csv(
        os.path.join(out_dir, f"fusion_benefit_perimg_{domain_shift}_seed{seed}.csv"),
        index=False,
    )

    n_images = len(per_img_df)
    n_images_with_dis = int(per_img_df["B_i"].notna().sum())

    return {
        "domain_shift": domain_shift, "seed": seed,
        "B_macro": per_img_df["B_i"].mean(skipna=True),
        "B_macro_std": per_img_df["B_i"].std(skipna=True),
        "B_pooled": b_pooled,
        "mean_coverage_C_D": per_img_df["coverage_C_D"].mean(),
        "n_images": n_images,
        "n_images_with_disagreement": n_images_with_dis,
        "n_disagreement_pixels_total": Ndis,
        "N10_total": N10, "N01_total": N01,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_shifts", nargs="+", default=list(ALL_SHIFTS.keys()),
                         choices=list(ALL_SHIFTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_root", default="/kaggle/working/outputs/full_checkpoints")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/fusion_benefit")
    args = parser.parse_args()

    rows = []
    for shift in args.domain_shifts:
        for seed in args.seeds:
            print(f"[{shift} seed={seed}] dang tinh B...")
            row = run_shift_seed(shift, seed, args.device, args.ckpt_root, args.out_dir)
            rows.append(row)
            print(f"  B_macro={row['B_macro']:.4f}  B_pooled={row['B_pooled']:.4f}  "
                  f"coverage={row['mean_coverage_C_D']:.4f}  "
                  f"n_img_with_dis={row['n_images_with_disagreement']}/{row['n_images']}")

    summary = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "fusion_benefit_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Aggregate CAP SHIFT (n=4, KHONG phai n=12) — dung nguyen tac da khoa
    shift_agg = summary.groupby("domain_shift").agg(
        B_macro_mean_over_seeds=("B_macro", "mean"),
        B_macro_std_over_seeds=("B_macro", "std"),
        B_pooled_mean_over_seeds=("B_pooled", "mean"),
        mean_coverage=("mean_coverage_C_D", "mean"),
    ).reset_index()
    shift_agg["Y_known_fusion_benefit"] = shift_agg["domain_shift"].map(KNOWN_FUSION_BENEFIT_Y)

    shift_agg_path = os.path.join(args.out_dir, "fusion_benefit_shift_level.csv")
    shift_agg.to_csv(shift_agg_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 120)
    print("FUSION BENEFIT AUDIT — B (directional disagreement symmetry), n=4 SHIFT (khong phai n=12)")
    print("=" * 120)
    print(shift_agg.to_string(index=False))
    print(f"\nChi tiet per-seed: {summary_path}")
    print(f"Aggregate cap shift: {shift_agg_path}")

    print("\nDIEN GIAI (chi mo ta, KHONG threshold, KHONG phan loai):")
    print("  Y_known_fusion_benefit=1 nghia la Dice_mean > Dice_best_single (da biet tu truoc,")
    print("  CHI de doi chieu, khong dung trong tinh B).")
    print("  Cau hoi duy nhat: C->R (Y=1) co B_macro noi bat RO RET va ON DINH hon 3 shift")
    print("  con lai (Y=0) khong? Neu co -> development observation dang chu y (chua phai")
    print("  gate da kiem chung). Neu khong -> dong huong automatic gate tren 4 shift nay,")
    print("  chuyen sang viet khoa luan voi ket qua pipeline hien co.")
    print("\nKHONG tao B2/composite/weighted-B sau khi xem ket qua nay.")


if __name__ == "__main__":
    main()
