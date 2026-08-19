"""
probability_geometry_audit.py

MUC DICH — audit CO CHE, khong phai gate. fusion_benefit_audit.py (B =
directional symmetry) da FAIL va DAO NGUOC ky vong: C->R (fusion co loi
nhat) lai co B THAP NHAT (disagreement gan nhu 1 chieu: CLAHE them
foreground, raw hau nhu khong them gi nguoc lai). Vay cau hoi dung khong
phai "huong bat dong co can bang khong", ma la:

    Trong MOT huong bat dong gan nhu tuyet doi, soft probability mean co
    tao duoc mot PHAN TACH CHON LOC (selective partition) hay khong —
    tuc mean co "giu lai" dung nhung CLAHE-addition la mach that, va
    "loai bo" dung nhung CLAHE-addition la nhieu, hay chi don gian la
    copy toan bo/khong gi ca?

KHONG train, KHONG tune, KHONG tao composite score. Chi 2 loai output:
    (a) MO TA hinh hoc xac suat trong Omega01/Omega10 — KHONG dung GT.
    (b) AUDIT HAU NGHIEM bang GT — CHI de tra loi "giu dung bao nhieu,
        loai dung bao nhieu", KHONG dung de chon/sua bat ky thanh phan
        nao cua (a) hay cua method.

DINH NGHIA:
    Omega01 = {p_raw < 0.5, p_clahe >= 0.5}   (CLAHE THEM foreground)
    Omega10 = {p_raw >= 0.5, p_clahe < 0.5}   (raw THEM foreground)
    p_mean  = (p_raw + p_clahe) / 2
    "giu lai" (survive) = p_mean >= 0.5 tai pixel do (mean van cho la
        foreground du 1 ben khong dong y)
    "dung" khi giu lai  = GT=1 (foreground that)
    "dung" khi loai bo  = GT=0 (background that)

CHAY (trong TT_SFUDA_2D/, sau khi da co 24 checkpoint full-budget):
    python run.py probability_geometry_audit.py \
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
MARGIN_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


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


def region_stats(p_r, p_c, p_mean, mask_mean, region_mask, gt_bool, suffix):
    """Mo ta + audit cho 1 vung (Omega01 hoac Omega10). Tra ve dict rong
    (NaN) neu region rong — khong set 0."""
    keys = [
        f"survive_rate_{suffix}", f"median_pr_{suffix}", f"median_pc_{suffix}",
        f"median_pmean_{suffix}", f"iqr_pmean_{suffix}",
        f"precision_kept_{suffix}", f"precision_rejected_{suffix}",
        f"n_kept_{suffix}", f"n_rejected_{suffix}",
    ]
    n_region = int(region_mask.sum())
    if n_region == 0:
        return {k: float("nan") for k in keys}

    survive = mask_mean[region_mask]
    kept = region_mask & mask_mean
    rejected = region_mask & (~mask_mean)

    pmean_region = p_mean[region_mask]
    q75, q25 = np.percentile(pmean_region, [75, 25])

    out = {
        f"survive_rate_{suffix}": float(survive.mean()),
        f"median_pr_{suffix}": float(np.median(p_r[region_mask])),
        f"median_pc_{suffix}": float(np.median(p_c[region_mask])),
        f"median_pmean_{suffix}": float(np.median(pmean_region)),
        f"iqr_pmean_{suffix}": float(q75 - q25),
        f"n_kept_{suffix}": int(kept.sum()),
        f"n_rejected_{suffix}": int(rejected.sum()),
    }
    out[f"precision_kept_{suffix}"] = (
        float(gt_bool[kept].mean()) if kept.sum() > 0 else float("nan")
    )
    out[f"precision_rejected_{suffix}"] = (
        float((~gt_bool[rejected]).mean()) if rejected.sum() > 0 else float("nan")
    )
    return out


def compute_geometry(raw_model, clahe_model, cfg, target, device):
    raw_loader, img_ids = get_loader(cfg, target, "test", use_clahe=False)
    clahe_loader, img_ids_c = get_loader(cfg, target, "test", use_clahe=True)
    assert img_ids == img_ids_c, "Thu tu anh raw/clahe test loader khong khop."

    rows = []
    for idx, (batch_raw, batch_clh) in enumerate(zip(raw_loader, clahe_loader)):
        inp_raw, tgt = batch_raw[0], batch_raw[1]
        inp_clh = batch_clh[0]
        gt_bool = (tgt.numpy().squeeze() >= 0.5)

        p_r = model_forward(raw_model, inp_raw, device)
        p_c = model_forward(clahe_model, inp_clh, device)
        p_mean = (p_r + p_c) / 2.0

        mask_r = p_r >= THRESHOLD
        mask_c = p_c >= THRESHOLD
        mask_mean = p_mean >= THRESHOLD

        omega01 = (~mask_r) & mask_c
        omega10 = mask_r & (~mask_c)
        n01, n10 = int(omega01.sum()), int(omega10.sum())
        n_dis = n01 + n10
        n_px = mask_r.size

        row = {
            "image": img_ids[idx],
            "coverage_disagree": n_dis / n_px,
            "n01": n01, "n10": n10,
            "frac_omega01": (n01 / n_dis) if n_dis > 0 else float("nan"),
            "frac_omega10": (n10 / n_dis) if n_dis > 0 else float("nan"),
        }
        row.update(region_stats(p_r, p_c, p_mean, mask_mean, omega01, gt_bool, "01"))
        row.update(region_stats(p_r, p_c, p_mean, mask_mean, omega10, gt_bool, "10"))

        dis_mask = omega01 | omega10
        if n_dis > 0:
            margin = np.abs(p_mean[dis_mask] - 0.5)
            hist, _ = np.histogram(margin, bins=MARGIN_BINS)
            for i in range(len(MARGIN_BINS) - 1):
                row[f"frac_margin_bin{i}"] = float(hist[i] / n_dis)
        else:
            for i in range(len(MARGIN_BINS) - 1):
                row[f"frac_margin_bin{i}"] = float("nan")

        rows.append(row)
    return pd.DataFrame(rows)


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

    df = compute_geometry(raw_model, clahe_model, cfg, target, device)
    df["domain_shift"] = domain_shift
    df["seed"] = seed

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"geometry_perimg_{domain_shift}_seed{seed}.csv"), index=False)

    macro_cols = [c for c in df.columns if c not in ("image", "domain_shift", "seed")]
    summary_row = {"domain_shift": domain_shift, "seed": seed}
    for c in macro_cols:
        summary_row[c] = df[c].mean(skipna=True)
    return summary_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_shifts", nargs="+", default=list(ALL_SHIFTS.keys()),
                         choices=list(ALL_SHIFTS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_root", default="/kaggle/working/outputs/full_checkpoints")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/prob_geometry")
    args = parser.parse_args()

    rows = []
    for shift in args.domain_shifts:
        for seed in args.seeds:
            print(f"[{shift} seed={seed}] dang audit probability geometry...")
            row = run_shift_seed(shift, seed, args.device, args.ckpt_root, args.out_dir)
            rows.append(row)
            print(f"  frac_omega01={row['frac_omega01']:.3f}  survive_rate_01={row['survive_rate_01']:.3f}  "
                  f"precision_kept_01={row['precision_kept_01']:.3f}  "
                  f"precision_rejected_01={row['precision_rejected_01']:.3f}")

    summary = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "geometry_summary_perseed.csv")
    summary.to_csv(summary_path, index=False)

    key_cols = [
        "domain_shift", "coverage_disagree", "frac_omega01", "frac_omega10",
        "survive_rate_01", "precision_kept_01", "precision_rejected_01",
        "survive_rate_10", "precision_kept_10", "precision_rejected_10",
    ]
    shift_agg = summary.groupby("domain_shift")[[c for c in key_cols if c != "domain_shift"]].mean().reset_index()
    shift_agg_path = os.path.join(args.out_dir, "geometry_summary_shift_level.csv")
    shift_agg.to_csv(shift_agg_path, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 130)
    print("PROBABILITY GEOMETRY AUDIT — cap SHIFT (trung binh qua seed)")
    print("=" * 130)
    print(shift_agg.to_string(index=False))
    print(f"\nChi tiet per-seed: {summary_path}")
    print(f"Per-image (day du cot, dung cho phan tich sau): {args.out_dir}/geometry_perimg_<shift>_seed<n>.csv")

    print("\nDoc ket qua (mo ta co che, KHONG phai gate/score):")
    print("  - precision_kept_01 CAO + precision_rejected_01 CAO dong thoi")
    print("    => mean dang PHAN TACH CHON LOC that (giu dung, loai dung),")
    print("    khong phai chi copy CLAHE hay chi copy raw.")
    print("  - precision_kept_01 ~ precision_rejected_01 ~ thap/random")
    print("    => mean khong phan tach duoc gi, gain Dice (neu co) den tu")
    print("    nguon khac, khong phai selective soft-filtering.")
    print("  - So sanh giua C->R (fusion co loi) va 3 shift con lai (fusion")
    print("    co hai) o dung 2 cot precision_kept_01/precision_rejected_01")
    print("    la trong tam de doc, khong phai coverage hay frac_omega.")
    print("\nKHONG tao composite score tu cac cot nay. Day la audit co che,")
    print("buoc tiep theo (logit mean / reliability-weighted) chi nen lam")
    print("SAU KHI da hieu ro co che nay, khong phai thay the cho no.")


if __name__ == "__main__":
    main()
