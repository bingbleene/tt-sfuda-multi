"""
tier4_reliability_curve.py

MUC DICH (Tier 4 — audit separability, KHONG train, KHONG chon threshold
cuoi cung cho method)
------------------------------------------------------------------
Tier 3 chung minh complementarity ceiling that (oracle fusion Dice cao
hon best individual ~0.11-0.14). Tier 3.5 chung minh R_flip (mot minh)
dinh vi duoc teacher dung ~0.68-0.71 (random baseline 0.50), on dinh hon
R_temp/R_combo.

Tier 4 tra loi cau hoi tiep theo: NEU chi cho router "dam chon" o vung
reliability-gap du lon (|R_r - R_c| > tau), no co dat duoc PRECISION cao
o mot vung COVERAGE con hop ly khong? Day la selective prediction, khac
voi accuracy toan bo vung bat dong ma Tier 3.5 da do.

Output CHINH la precision-coverage curve (nhieu diem theo tau), KHONG
phai 1 con so PASS/FAIL. Quyet dinh co build router hay khong dua tren
DOC curve nay, khong dua tren 1 threshold dinh san.

RANH GIOI QUAN TRONG NHAT:
    - GT chi dung de CHAM DIEM precision tai moi tau — KHONG dung de chon
      tau, khong dung de chon trong so composite. Luoi tau va cong thuc
      composite deu KHOA CUNG TRUOC KHI CHAY, ghi trong code nay, khong
      duoc sua sau khi xem ket qua.
    - Tau cuoi cung dung cho method that (neu co) PHAI dat bang 1 rule
      khong nhan (vd quantile cua reliability gap, hoac coverage target
      co dinh) — KHONG duoc chon tau tai diem co precision GT dep nhat
      tren chinh 2 shift pilot nay, do la leakage.

2 CANDIDATE DA KHOA (khong doi sau khi chay):
    R_flip(i)      = 1 - U_tta(i),  U_tta(i) = |p(x)_i - flip(p(flip(x)))_i|
        — DA kiem chung o Tier 3.5, ~0.68-0.71 accuracy, on dinh nhat.
    R_composite(i) = [ z(-H) + z(-U_tta) + z(-U_temp) ] / 3
        — z-score TUNG THANH PHAN THEO TUNG ANH (khong theo toan tap),
          trong so BANG NHAU (1/3 moi thanh phan) — khong tune.
        H(i)      = -p*log(p) - (1-p)*log(1-p)      (entropy pixel)
        U_temp(i) = |p_epoch_hien_tai(i) - p_epoch_truoc(i)|
        LUU Y: tai epoch=0, khong co "epoch truoc" that (epoch_prev=0,
        trung voi epoch hien tai) -> U_temp = 0 MOI NOI -> z-score suy
        bien (std=0) -> thanh phan nay dong gop 0 vao composite tai
        epoch=0 (composite epoch=0 thuc chat chi con trung binh cua
        z(-H) va z(-U_tta)). Day la gioi han ky thuat da biet truoc,
        khong phai loi.

LUOI THRESHOLD DA KHOA (khong them/bot diem sau khi xem curve):
    TAUS = np.arange(0.0, 0.51, 0.05)   # 11 diem: 0.00, 0.05, ..., 0.50

METRIC MOI TAU, MOI (shift, epoch, seed, candidate):
    precision_micro   = tong so pixel chon dung / tong so pixel duoc chon
                         (pool toan bo pixel, khong theo anh)
    precision_macro   = trung binh precision TUNG ANH (chi anh co pixel
                         duoc chon), tranh anh nhieu foreground lan at
    coverage_micro    = tong so pixel duoc chon / tong so pixel bat dong
    coverage_macro    = trung binh (so pixel chon / so pixel bat dong)
                         TUNG ANH (chi anh co pixel bat dong)
    n_disagreement_pixels, n_selected_pixels,
    n_images_total, n_images_selected
                      — de phat hien curve dep gia do coverage/so anh qua
                        nho (dac biet hrf2chase chi co 8 anh test)

Baseline ngau nhien trong vung bat dong = 0.50 (da chung minh o Tier 3:
GT nhi phan, dung 1 trong 2 teacher, khong co "ca hai sai" tai day).

CHAY (trong TT_SFUDA_2D/, sau Tier 3.5, tai su dung checkpoint da co):
    python run.py tier4_reliability_curve.py \
        --domain_shifts chase2rite hrf2chase \
        --tier2_ckpt_dir <duong dan checkpoint Tier 2>
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
# CONFIG — KHOA CUNG, khong doi sau khi chay
# ============================================================

PILOT_SHIFTS = {"chase2rite": 768, "hrf2chase": 384}
SEEDS = ["A", "B"]
EPOCHS = [0, 3]
EPOCH_PREV_MAP = {0: 0, 3: 2}

THRESHOLD = 0.5
EPS = 1e-7
TAUS = np.round(np.arange(0.0, 0.51, 0.05), 2)


# ============================================================
# Model / preprocess
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


def predict_prob_flip_tta(model, img_bgr: np.ndarray, resolution: int, device: str) -> np.ndarray:
    img_flipped = cv2.flip(img_bgr, 1)
    prob_flipped = predict_prob(model, img_flipped, resolution, device)
    return cv2.flip(prob_flipped, 1)


def get_test_image_mask_paths(target: str, cfg: dict):
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


def entropy_map(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, EPS, 1 - EPS)
    return -(p_c * np.log(p_c) + (1 - p_c) * np.log(1 - p_c))


def zscore_per_image(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    if std < EPS:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def compute_teacher_signals(model_cur, model_prev, img_bgr, resolution, device, target_hw):
    h, w = target_hw
    p_cur = cv2.resize(predict_prob(model_cur, img_bgr, resolution, device), (w, h),
                        interpolation=cv2.INTER_LINEAR)
    p_flip = cv2.resize(predict_prob_flip_tta(model_cur, img_bgr, resolution, device), (w, h),
                         interpolation=cv2.INTER_LINEAR)
    p_prev = cv2.resize(predict_prob(model_prev, img_bgr, resolution, device), (w, h),
                         interpolation=cv2.INTER_LINEAR)

    u_tta = np.abs(p_cur - p_flip)
    u_temp = np.abs(p_cur - p_prev)
    h_map = entropy_map(p_cur)

    r_flip = 1.0 - u_tta
    r_composite = (zscore_per_image(-h_map) + zscore_per_image(-u_tta) +
                   zscore_per_image(-u_temp)) / 3.0

    return p_cur, r_flip, r_composite


def sweep_thresholds(r_raw: np.ndarray, r_clh: np.ndarray, raw_correct: np.ndarray,
                      disagree: np.ndarray, taus: np.ndarray) -> list:
    gap = r_raw - r_clh
    rows = []
    for tau in taus:
        selected = disagree & (np.abs(gap) > tau)
        n_selected = int(selected.sum())
        if n_selected == 0:
            rows.append({"tau": float(tau), "n_selected": 0, "n_correct": 0})
            continue
        chosen_raw = gap[selected] > 0
        correct = np.where(chosen_raw, raw_correct[selected], ~raw_correct[selected])
        rows.append({"tau": float(tau), "n_selected": n_selected, "n_correct": int(correct.sum())})
    return rows


def run_shift(domain_shift: str, resolution: int, device: str, tier2_ckpt_dir: str, out_dir: str):
    cfg = load_config(domain_shift)
    _, target = domain_shift.split("2")
    img_mask_pairs = get_test_image_mask_paths(target, cfg)
    n_images_total = len(img_mask_pairs)
    print(f"[{domain_shift}] {n_images_total} cap anh/mask (test set)")

    def ckpt(view_seed, epoch):
        p = os.path.join(tier2_ckpt_dir, domain_shift, view_seed, f"epoch_{epoch}.pth")
        if not os.path.exists(p):
            raise RuntimeError(f"Thieu checkpoint: {p}")
        return p

    def load_model(view_seed, epoch):
        m = load_arch(cfg, device)
        load_checkpoint_into(m, ckpt(view_seed, epoch), device)
        return m

    all_rows = []

    for seed in SEEDS:
        for epoch in EPOCHS:
            epoch_prev = EPOCH_PREV_MAP[epoch]
            if epoch_prev == epoch:
                print(f"[{domain_shift} seed={seed} epoch={epoch}] "
                      f"CANH BAO: khong co epoch truoc that, U_temp=0 moi noi.")

            raw_cur = load_model(f"raw_{seed}", epoch)
            raw_prev = load_model(f"raw_{seed}", epoch_prev)
            clahe_cur = load_model(f"clahe_{seed}", epoch)
            clahe_prev = load_model(f"clahe_{seed}", epoch_prev)

            per_image_data = []
            for img_path, mask_path in img_mask_pairs:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                h, w = img.shape[:2]
                clh = apply_clahe_lab(img)
                gt_bool = gt > 127

                p_raw, rflip_raw, rcomp_raw = compute_teacher_signals(
                    raw_cur, raw_prev, img, resolution, device, (h, w))
                p_clh, rflip_clh, rcomp_clh = compute_teacher_signals(
                    clahe_cur, clahe_prev, clh, resolution, device, (h, w))

                mask_raw = p_raw >= THRESHOLD
                mask_clh = p_clh >= THRESHOLD
                disagree = mask_raw != mask_clh
                raw_correct = mask_raw == gt_bool

                per_image_data.append({
                    "image": os.path.basename(img_path),
                    "disagree": disagree,
                    "raw_correct": raw_correct,
                    "n_disagree": int(disagree.sum()),
                    "R_flip": (rflip_raw, rflip_clh),
                    "R_composite": (rcomp_raw, rcomp_clh),
                })

            for candidate in ["R_flip", "R_composite"]:
                for tau in TAUS:
                    total_selected = 0
                    total_correct = 0
                    total_disagree = 0
                    per_image_precisions = []
                    per_image_coverages = []
                    n_images_selected = 0

                    for d in per_image_data:
                        r_raw, r_clh = d[candidate]
                        result = sweep_thresholds(r_raw, r_clh, d["raw_correct"], d["disagree"],
                                                   np.array([tau]))[0]
                        n_sel = result["n_selected"]
                        n_cor = result["n_correct"]
                        n_dis = d["n_disagree"]

                        total_selected += n_sel
                        total_correct += n_cor
                        total_disagree += n_dis

                        if n_sel > 0:
                            n_images_selected += 1
                            per_image_precisions.append(n_cor / n_sel)
                        if n_dis > 0:
                            per_image_coverages.append(n_sel / n_dis)

                    precision_micro = (total_correct / total_selected) if total_selected > 0 else float("nan")
                    precision_macro = (float(np.mean(per_image_precisions))
                                        if per_image_precisions else float("nan"))
                    coverage_micro = (total_selected / total_disagree) if total_disagree > 0 else float("nan")
                    coverage_macro = (float(np.mean(per_image_coverages))
                                       if per_image_coverages else float("nan"))

                    all_rows.append({
                        "domain_shift": domain_shift, "epoch": epoch, "seed": seed,
                        "router": candidate, "tau": tau,
                        "precision_micro": precision_micro, "precision_macro": precision_macro,
                        "coverage_micro": coverage_micro, "coverage_macro": coverage_macro,
                        "n_disagreement_pixels": total_disagree,
                        "n_selected_pixels": total_selected,
                        "n_images_total": n_images_total,
                        "n_images_selected": n_images_selected,
                    })

            print(f"[{domain_shift} seed={seed} epoch={epoch}] xong.")

    df = pd.DataFrame(all_rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"tier4_curve_{domain_shift}.csv")
    df.to_csv(csv_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_shifts", nargs="+", default=list(PILOT_SHIFTS.keys()),
                         choices=list(PILOT_SHIFTS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tier2_ckpt_dir", default="/kaggle/working/outputs/tier2_survival")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/tier4")
    args = parser.parse_args()

    all_df = []
    for shift in args.domain_shifts:
        resolution = PILOT_SHIFTS[shift]
        df = run_shift(shift, resolution, args.device, args.tier2_ckpt_dir, args.out_dir)
        all_df.append(df)

    full_df = pd.concat(all_df, ignore_index=True)
    full_path = os.path.join(args.out_dir, "tier4_curve_all.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    full_df.to_csv(full_path, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)

    print("\n" + "=" * 120)
    print("TIER 4 — Precision-Coverage Curve (baseline ngau nhien trong vung bat dong = 0.50)")
    print("=" * 120)
    for shift in args.domain_shifts:
        for candidate in ["R_flip", "R_composite"]:
            sub = full_df[(full_df["domain_shift"] == shift) & (full_df["router"] == candidate)]
            print(f"\n--- {shift} / {candidate} ---")
            print(sub[["epoch", "seed", "tau", "precision_micro", "precision_macro",
                        "coverage_micro", "coverage_macro", "n_selected_pixels",
                        "n_images_selected", "n_images_total"]].to_string(index=False))

    print(f"\nChi tiet: {full_path}")
    print("\nDoc ket qua theo dung 3 dieu kien da chot (khong PASS/FAIL scalar don):")
    print("  1) precision_micro/macro > 0.50 ro ret o vung tau vua phai")
    print("  2) coverage khong sup ve gan 0 tai vung precision cao do")
    print("  3) pattern giu duoc qua CA 4 trajectory: C->R seed A/B, H->C seed A/B,")
    print("     epoch 0 va epoch 3 — khong chi dep o 1 truong hop")
    print("  Chu y rieng cho hrf2chase (chi 8 anh test): n_images_selected thap")
    print("  o tau lon la dau hieu curve dang duoc dung boi rat it anh, doc than")
    print("  trong khi trich dan.")
    print("\nKHONG dung ket qua nay de chon tau cuoi cung cho method — tau that")
    print("phai dat bang 1 rule khong nhan (quantile cua reliability gap, hoac")
    print("coverage target co dinh), quyet dinh o buoc khac, khong o file nay.")


if __name__ == "__main__":
    main()
