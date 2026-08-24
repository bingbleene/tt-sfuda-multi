"""
tier1_frangi_enrichment.py

MUC DICH (Tier 1 — van chua train gi ca)
------------------------------------------
Tier 0 da cho thay: D_fg_union lon (0.45-0.58) NHUNG fg_ratio_change cung
lon va nhat quan ca 4 shift (+83% den +139%) — trung ho hanh vi da tung
gay -10 den -12 diem Dice khi ep CLAHE hard truoc day. Tier 1 tra loi:
phan foreground CLAHE "them vao" do la vessel-like recovery hay background
explosion — dung Frangi lam STRUCTURAL REFEREE (khong phai ground-truth
proxy, khong dung de sua prediction).

TAI SU DUNG TOAN BO tu unsupervised_selector_v5b.py (KHONG viet lai logic):
    load_model, apply_clahe_lab, predict, frangi_map, recovery_metrics,
    resize01, summarize
Script nay CHI doi 1 cho so voi ban goc trong unsupervised_selector_v5b.py:
    predict(...) chay o resolution DA FREEZE cho tung shift, thay vi
    canonical=512 co dinh.

METRIC TAI SU DUNG (dung nguyen recovery_metrics, khong sua):
    recovered_ratio, recovered_vessel_like, recovered_stable_fraction,
    stable_recovery_ratio, background_explosion, fg_raw, fg_clahe, fg_gain

METRIC MOI THEM (Tier 1 rieng, khong co trong ban goc):
    D_mean_frangi_high / D_mean_frangi_low — trung binh |p_raw-p_clahe|
        trong top-10% Frangi-response so voi 90% con lai
    enrichment_ratio_R_F = D_mean_frangi_high / D_mean_frangi_low
        R_F >> 1 nghia la disagreement TAP TRUNG o vung vessel-like
    spearman_D_F — Spearman rank correlation giua D_i va F_i (thay Pearson
        vi phan phoi qua lech), chi la tin hieu phu, KHONG dung lam metric
        duy nhat (dung theo dung gop y da thong nhat)

RANH GIOI: KHONG doc target mask/label o bat ky dong nao. Anh target lay
tu inputs/{target}/train/images (giong dung unsupervised_selector_v5b.py).

CHAY (trong TT_SFUDA_2D/, sau khi unsupervised_selector_v5b.py da ton tai
trong cung thu muc — script nay import truc tiep tu do):
    python run.py tier1_frangi_enrichment.py \
        --domain_shifts chase2hrf chase2rite hrf2chase hrf2rite

DOC KET QUA — dung tieu chi ban da chot:
    - C->R / H->R: recovered_vessel_like CAO, background_explosion THAP
      => tin hieu manh, dang xay dual-teacher (giai thich dung vi sao
      CLAHE giup RITE)
    - C->H / H->C: recovered_vessel_like THAP, background_explosion CAO
      => xac nhan canh bao Tier 0, KHONG dung CLAHE view cho 2 shift nay
    - Ca 4 shift cho pattern gan giong nhau
      => Frangi Tier 1 khong phan biet duoc domain, KHONG dung lam nen
      cho dual-teacher, quay lai suy nghi huong khac truoc khi ban Tier 2
"""

import os
import argparse
from glob import glob

import cv2
import numpy as np
import pandas as pd
from scipy import stats
import torch

from unsupervised_selector_v5b import (
    load_model,
    apply_clahe_lab,
    predict,
    frangi_map,
    recovery_metrics,
    resize01,
    summarize,
)

# ============================================================
# CONFIG
# ============================================================

FROZEN_RESOLUTION = {
    "chase2hrf": 1024,
    "chase2rite": 768,
    "hrf2chase": 384,
    "hrf2rite": 512,
}

THRESHOLD = 0.5
RECOVERY_DELTA_DEFAULT = 0.15    # giu dung default cua V3/V5b
FRANGI_THRESHOLD_DEFAULT = 0.01  # giu dung default cua V3/V5b
DIAG_SIZE_DEFAULT = 512          # kich thuoc chung de so sanh metric,
                                  # doc lap voi resolution du doan
FRANGI_HIGH_PERCENTILE = 90      # top 10% Frangi-response
SPEARMAN_SAMPLE = 20000          # subsample pixel de Spearman khong cham
EPS = 1e-7


# ============================================================
# Enrichment metrics — MOI, chua co trong unsupervised_selector_v5b.py
# ============================================================

def enrichment_metrics(D_i: np.ndarray, F_i: np.ndarray, rng: np.random.RandomState) -> dict:
    thresh = np.percentile(F_i, FRANGI_HIGH_PERCENTILE)
    high_mask = F_i >= thresh
    low_mask = ~high_mask

    d_high = float(D_i[high_mask].mean()) if high_mask.sum() > 0 else float("nan")
    d_low = float(D_i[low_mask].mean()) if low_mask.sum() > 0 else float("nan")
    r_f = d_high / (d_low + EPS)

    flat_d = D_i.flatten()
    flat_f = F_i.flatten()
    n_sample = min(SPEARMAN_SAMPLE, flat_d.size)
    idx = rng.choice(flat_d.size, size=n_sample, replace=False)
    rho, pval = stats.spearmanr(flat_d[idx], flat_f[idx])

    return {
        "D_mean_frangi_high": d_high,
        "D_mean_frangi_low": d_low,
        "enrichment_ratio_R_F": r_f,
        "spearman_D_F": float(rho),
        "spearman_D_F_pval": float(pval),
    }


# ============================================================
# Main loop per shift
# ============================================================

def run_for_shift(
    domain_shift: str,
    device: str,
    out_dir: str,
    max_images,
    recovery_delta: float,
    frangi_threshold: float,
    diag_size: int,
    seed: int,
):
    source_domain, target = domain_shift.split("2")
    source = f"{source_domain}_unet"
    resolution = FROZEN_RESOLUTION[domain_shift]

    model, cfg = load_model(source, target, device)

    img_dir = os.path.join("inputs", target, "train", "images")
    paths = sorted(glob(os.path.join(img_dir, "*" + cfg["img_ext"])))
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise RuntimeError(f"Khong tim thay anh target: {img_dir}")

    rng = np.random.RandomState(seed)
    rows = []

    for i, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Khong doc duoc anh: {path}")

        fr = frangi_map(img, diag_size)

        raw = img
        clh = apply_clahe_lab(img)
        raw_f = cv2.flip(raw, 1)
        clh_f = cv2.flip(clh, 1)

        p_raw = resize01(predict(model, raw, resolution, device), diag_size)
        p_clh = resize01(predict(model, clh, resolution, device), diag_size)
        pf_raw = resize01(cv2.flip(predict(model, raw_f, resolution, device), 1), diag_size)
        pf_clh = resize01(cv2.flip(predict(model, clh_f, resolution, device), 1), diag_size)

        rec = recovery_metrics(
            p_raw, p_clh, pf_raw, pf_clh, fr,
            THRESHOLD, recovery_delta, frangi_threshold,
        )

        d_i = np.abs(p_raw - p_clh)
        enr = enrichment_metrics(d_i, fr, rng)

        row = {**rec, **enr, "image": os.path.basename(path), "domain_shift": domain_shift}
        rows.append(row)
        print(f"[{domain_shift} {i+1}/{len(paths)}] {os.path.basename(path)}")

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"tier1_frangi_{domain_shift}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[{domain_shift}] {len(df)} anh -> {csv_path}")
    return df


def summarize_shift(df: pd.DataFrame, domain_shift: str) -> dict:
    metric_cols = [
        "recovered_ratio", "recovered_vessel_like", "recovered_stable_fraction",
        "stable_recovery_ratio", "background_explosion",
        "fg_raw", "fg_clahe", "fg_gain",
        "enrichment_ratio_R_F", "spearman_D_F",
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
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/tier1_frangi")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--recovery_delta", type=float, default=RECOVERY_DELTA_DEFAULT)
    parser.add_argument("--frangi_threshold", type=float, default=FRANGI_THRESHOLD_DEFAULT)
    parser.add_argument("--diag_size", type=int, default=DIAG_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    all_summaries = []
    for shift in args.domain_shifts:
        df = run_for_shift(
            shift, args.device, args.out_dir, args.max_images,
            args.recovery_delta, args.frangi_threshold, args.diag_size, args.seed,
        )
        all_summaries.append(summarize_shift(df, shift))

    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(args.out_dir, "tier1_summary_all_shifts.csv")
    summary_df.to_csv(summary_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    print("\n" + "=" * 110)
    print("TIER 1 — Frangi structural referee (KHONG doc nhan target)")
    print(f"recovery_delta={args.recovery_delta}  frangi_threshold={args.frangi_threshold}  diag_size={args.diag_size}")
    print("=" * 110)
    print(summary_df[[
        "domain_shift", "n_images",
        "recovered_vessel_like_mean", "recovered_vessel_like_std",
        "background_explosion_mean", "background_explosion_std",
        "stable_recovery_ratio_mean",
        "fg_gain_mean",
        "enrichment_ratio_R_F_mean",
        "spearman_D_F_mean",
    ]].to_string(index=False))

    print(f"\nBang chi tiet: {summary_path}")
    print("\nDoc ket qua (dung tieu chi da chot, khong dung threshold cung):")
    print("  - C->R / H->R: recovered_vessel_like CAO + background_explosion THAP")
    print("    => tin hieu manh, giai thich dung vi sao CLAHE tung giup RITE")
    print("  - C->H / H->C: recovered_vessel_like THAP + background_explosion CAO")
    print("    => xac nhan canh bao Tier 0 (fg_ratio_change lon o Tier 0 chu yeu")
    print("    la artefact, khop voi thi nghiem CLAHE-hard truoc day gay -10 den")
    print("    -12 diem Dice)")
    print("  - enrichment_ratio_R_F >> 1 o shift nao => disagreement o shift do")
    print("    tap trung vung vessel-like that, khong phai noise rai rac")
    print("  - Neu ca 4 shift cho pattern gan giong nhau => Frangi Tier 1 khong")
    print("    phan biet duoc domain, KHONG dung lam nen cho dual-teacher")
    print("\nLuu y: day van la diagnostic khong train. Neu Tier 1 cho tin hieu")
    print("phan hoa ro giua cac shift, buoc tiep theo moi la Tier 2 (control")
    print("training + stochastic control raw_A/raw_B, clahe_A/clahe_B).")


if __name__ == "__main__":
    main()
