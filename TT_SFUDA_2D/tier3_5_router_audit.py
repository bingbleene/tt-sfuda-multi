"""
tier3_5_router_audit.py

MUC DICH (Tier 3.5 — audit rieng, tai su dung checkpoint + GT da co,
KHONG train them gi)
------------------------------------------------------------------
Tier 3 da chung minh complementarity ceiling that (G_comp ~0.11-0.14).
Tier 3.5 tra loi 2 cau hoi con lai TRUOC khi viet full architecture:

    (A) ROUTER AUDIT — Voi 4 cong thuc reliability KHOA CUNG TRUOC (khong
        tune sau khi nhin ket qua), tin hieu KHONG NHAN co dinh vi duoc
        teacher nao dung tai moi pixel bat dong hay khong? Neu tat ca
        ~50% (random), Tier 3 ceiling co that nhung KHONG khai thac duoc
        bang tin hieu unsupervised — dung xay router.

    (B) STOCHASTIC CONTROL cho G_comp — Tier 3 chi do G_comp(raw, clahe).
        Nhung oracle-pixel-fusion GAN NHU LUON DUONG khi ghep 2 model
        khong hoan hao, KE CA khi chung khong khac gi ve appearance (chi
        khac seed). Neu G_comp(raw_A, raw_B) (2 model CUNG view, khac
        seed) cung lon xap xi G_comp(raw, clahe), thi "complementarity"
        do KHONG dac thu appearance — chi la hieu ung ensemble chung
        chung, khong can CLAHE gi ca. Day la control con thieu tu Tier 3,
        tuong duong D_stoch da bat buoc o Tier 2.

RANH GIOI — Ke thua dung nguyen tac Tier 3:
    - GT CHI dung de CHAM DIEM 4 cong thuc da khoa san, KHONG dung de chon
      hay tune cong thuc nao — 4 cong thuc duoi day la TOAN BO candidate,
      khong duoc them/bot/chinh trong so sau khi thay ket qua.
    - Tai su dung checkpoint epoch_2 va epoch_3 da co san tu Tier 2
      (tier2_survival.py) — KHONG train them.

4 CONG THUC RELIABILITY DA KHOA (khong doi sau khi chay):
    R_conf(i) = |2*p(i) - 1|
        confidence — bien phu, KHONG dung mot minh (da biet entropy/
        confidence-only tung that bai o V1-V7 va response-recovery).
    R_flip(i)  = 1 - |p(x)_i - flip(p(flip(x)))_i|
        flip-TTA consistency — pixel on dinh qua phep doi xung ngang.
    R_temp(i)  = 1 - |p_epoch3(i) - p_epoch2(i)|
        "temporal" o MUC EPOCH (khong phai iteration-level) — CHI LA
        PROXY THO vi Tier 2 chi luu checkpoint moi epoch, khong phai moi
        buoc cap nhat. Ghi ro han che nay khi bao cao ket qua.
    R_combo(i) = R_flip(i) * R_temp(i)
        ket hop 2 tin hieu khong dung entropy-only.

METRIC CHINH:
    A_route(candidate) = P(argmax_k R_k(i) == teacher_dung(i) | pixel
                            bat dong, dung 1 teacher dung)
    Baseline ngau nhien = 50% (da chung minh o Tier 3: trong vung bat
    dong voi GT nhi phan, luon dung 1 trong 2, khong co "ca hai sai").

    G_comp_stochastic(raw_A, raw_B) va (clahe_A, clahe_B) — dung cong
    thuc G_comp giong het Tier 3 (oracle_fusion_dice - max(dice_A, dice_B))
    nhung ap cho 2 model CUNG VIEW, khac seed — de doi chieu truc tiep voi
    G_comp(raw, clahe) da co tu Tier 3.

CHAY (trong TT_SFUDA_2D/, sau Tier 3):
    python run.py tier3_5_router_audit.py \
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
# CONFIG
# ============================================================

PILOT_SHIFTS = {"chase2rite": 768, "hrf2chase": 384}
SEEDS = ["A", "B"]
THRESHOLD = 0.5
EPS = 1e-7
EPOCH_CURRENT = 3   # trang thai "final adapted" — diem van hanh that
EPOCH_PREV = 2      # dung cho R_temp (proxy epoch-level, xem canh bao tren)


# ============================================================
# Model / preprocess — dung nguyen ban da xac nhan Tier 0/1/2/3
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
    """Predict tren anh lat ngang, roi lat ket qua ve lai — dung cho R_flip."""
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


def dice_score(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = int((pred_bool & gt_bool).sum())
    denom = int(pred_bool.sum()) + int(gt_bool.sum())
    return 1.0 if denom == 0 else 2.0 * inter / denom


# ============================================================
# (A) Router audit — 4 cong thuc reliability da khoa
# ============================================================

def compute_reliability_signals(model, img_bgr, resolution, device, prev_epoch_model,
                                 target_hw):
    """Tra ve (p_current, R_conf, R_flip, R_temp) — tat ca resize ve target_hw."""
    h, w = target_hw
    p_cur = cv2.resize(predict_prob(model, img_bgr, resolution, device), (w, h),
                        interpolation=cv2.INTER_LINEAR)
    p_flip = cv2.resize(predict_prob_flip_tta(model, img_bgr, resolution, device), (w, h),
                         interpolation=cv2.INTER_LINEAR)
    p_prev = cv2.resize(predict_prob(prev_epoch_model, img_bgr, resolution, device), (w, h),
                         interpolation=cv2.INTER_LINEAR)

    r_conf = np.abs(2 * p_cur - 1)
    r_flip = 1.0 - np.abs(p_cur - p_flip)
    r_temp = 1.0 - np.abs(p_cur - p_prev)
    return p_cur, r_conf, r_flip, r_temp


def router_audit_pair(raw_cur, raw_prev, clahe_cur, clahe_prev, resolution, device, img_mask_pairs):
    rows = []
    for img_path, mask_path in img_mask_pairs:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        clh = apply_clahe_lab(img)
        gt_bool = gt > 127

        p_raw, rc_raw, rf_raw, rt_raw = compute_reliability_signals(
            raw_cur, img, resolution, device, raw_prev, (h, w))
        p_clh, rc_clh, rf_clh, rt_clh = compute_reliability_signals(
            clahe_cur, clh, resolution, device, clahe_prev, (h, w))

        mask_raw = p_raw >= THRESHOLD
        mask_clh = p_clh >= THRESHOLD
        disagree = mask_raw != mask_clh
        n_dis = int(disagree.sum())
        if n_dis == 0:
            continue

        raw_correct_dis = (mask_raw == gt_bool)[disagree]

        candidates = {
            "R_conf": (rc_raw, rc_clh),
            "R_flip": (rf_raw, rf_clh),
            "R_temp": (rt_raw, rt_clh),
            "R_combo": (rf_raw * rt_raw, rf_clh * rt_clh),
        }

        row = {"image": os.path.basename(img_path), "n_disagree_px": n_dis}
        for name, (r_raw, r_clh) in candidates.items():
            r_raw_dis = r_raw[disagree]
            r_clh_dis = r_clh[disagree]
            predicted_raw_better = r_raw_dis > r_clh_dis
            tie = r_raw_dis == r_clh_dis
            correct = np.where(predicted_raw_better, raw_correct_dis, ~raw_correct_dis)
            n_tie = int(tie.sum())
            n_valid = n_dis - n_tie
            acc = float(correct[~tie].sum() / n_valid) if n_valid > 0 else float("nan")
            row[f"A_route_{name}"] = acc
            row[f"tie_frac_{name}"] = n_tie / n_dis
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# (B) Stochastic control cho Complementarity Gain
# ============================================================

def g_comp_pair(model_1, model_2, resolution, device, img_mask_pairs):
    """G_comp giua 2 model CUNG mot appearance view — vd raw_A vs raw_B.
    Ca 2 model du doan tren CUNG 1 view (khong doi appearance) — chi khac
    seed huan luyen."""
    rows = []
    for img_path, mask_path in img_mask_pairs:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        gt_bool = gt > 127

        p1 = cv2.resize(predict_prob(model_1, img, resolution, device), (w, h),
                         interpolation=cv2.INTER_LINEAR)
        p2 = cv2.resize(predict_prob(model_2, img, resolution, device), (w, h),
                         interpolation=cv2.INTER_LINEAR)
        m1 = p1 >= THRESHOLD
        m2 = p2 >= THRESHOLD
        disagree = m1 != m2

        dice_1 = dice_score(m1, gt_bool)
        dice_2 = dice_score(m2, gt_bool)
        oracle_pred = np.where(disagree, gt_bool, m1)
        dice_oracle = dice_score(oracle_pred.astype(bool), gt_bool)
        g_comp = dice_oracle - max(dice_1, dice_2)

        rows.append({
            "image": os.path.basename(img_path),
            "dice_1": dice_1, "dice_2": dice_2,
            "dice_oracle_fusion": dice_oracle,
            "g_comp_stochastic": g_comp,
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

    def ckpt(view_seed, epoch):
        p = os.path.join(tier2_ckpt_dir, domain_shift, view_seed, f"epoch_{epoch}.pth")
        if not os.path.exists(p):
            raise RuntimeError(f"Thieu checkpoint: {p}")
        return p

    def load_model(view_seed, epoch):
        m = load_arch(cfg, device)
        load_checkpoint_into(m, ckpt(view_seed, epoch), device)
        return m

    router_rows = []
    gcomp_rows = []

    for seed in SEEDS:
        raw_cur = load_model(f"raw_{seed}", EPOCH_CURRENT)
        raw_prev = load_model(f"raw_{seed}", EPOCH_PREV)
        clahe_cur = load_model(f"clahe_{seed}", EPOCH_CURRENT)
        clahe_prev = load_model(f"clahe_{seed}", EPOCH_PREV)

        df_router = router_audit_pair(raw_cur, raw_prev, clahe_cur, clahe_prev,
                                       resolution, device, img_mask_pairs)
        df_router["domain_shift"] = domain_shift
        df_router["seed"] = seed
        router_rows.append(df_router)
        print(f"[ROUTER {domain_shift} seed={seed}] "
              f"A_conf={df_router['A_route_R_conf'].mean():.3f} "
              f"A_flip={df_router['A_route_R_flip'].mean():.3f} "
              f"A_temp={df_router['A_route_R_temp'].mean():.3f} "
              f"A_combo={df_router['A_route_R_combo'].mean():.3f}")

    # --- Stochastic control: raw_A vs raw_B, clahe_A vs clahe_B, tai epoch_3 ---
    raw_A = load_model("raw_A", EPOCH_CURRENT)
    raw_B = load_model("raw_B", EPOCH_CURRENT)
    clahe_A = load_model("clahe_A", EPOCH_CURRENT)
    clahe_B = load_model("clahe_B", EPOCH_CURRENT)

    df_gc_raw = g_comp_pair(raw_A, raw_B, resolution, device, img_mask_pairs)
    df_gc_raw["domain_shift"] = domain_shift
    df_gc_raw["pair"] = "raw_A_vs_raw_B"
    gcomp_rows.append(df_gc_raw)
    print(f"[G_COMP_STOCH {domain_shift} raw_A vs raw_B] "
          f"g_comp_stochastic={df_gc_raw['g_comp_stochastic'].mean():.4f}")

    df_gc_clahe = g_comp_pair(clahe_A, clahe_B, resolution, device, img_mask_pairs)
    df_gc_clahe["domain_shift"] = domain_shift
    df_gc_clahe["pair"] = "clahe_A_vs_clahe_B"
    gcomp_rows.append(df_gc_clahe)
    print(f"[G_COMP_STOCH {domain_shift} clahe_A vs clahe_B] "
          f"g_comp_stochastic={df_gc_clahe['g_comp_stochastic'].mean():.4f}")

    os.makedirs(out_dir, exist_ok=True)
    router_df = pd.concat(router_rows, ignore_index=True)
    gcomp_df = pd.concat(gcomp_rows, ignore_index=True)
    router_df.to_csv(os.path.join(out_dir, f"tier3_5_router_{domain_shift}.csv"), index=False)
    gcomp_df.to_csv(os.path.join(out_dir, f"tier3_5_gcomp_stochastic_{domain_shift}.csv"), index=False)
    return router_df, gcomp_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_shifts", nargs="+", default=list(PILOT_SHIFTS.keys()),
                         choices=list(PILOT_SHIFTS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tier2_ckpt_dir", default="/kaggle/working/outputs/tier2_survival")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/tier3_5")
    args = parser.parse_args()

    all_router, all_gcomp = [], []
    for shift in args.domain_shifts:
        resolution = PILOT_SHIFTS[shift]
        r_df, g_df = run_shift(shift, resolution, args.device, args.tier2_ckpt_dir, args.out_dir)
        all_router.append(r_df)
        all_gcomp.append(g_df)

    router_df = pd.concat(all_router, ignore_index=True)
    gcomp_df = pd.concat(all_gcomp, ignore_index=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    router_summary = router_df.groupby(["domain_shift", "seed"])[
        ["A_route_R_conf", "A_route_R_flip", "A_route_R_temp", "A_route_R_combo",
         "tie_frac_R_combo"]
    ].mean().reset_index()
    gcomp_summary = gcomp_df.groupby(["domain_shift", "pair"])["g_comp_stochastic"].mean().reset_index()

    os.makedirs(args.out_dir, exist_ok=True)
    router_summary.to_csv(os.path.join(args.out_dir, "tier3_5_router_summary.csv"), index=False)
    gcomp_summary.to_csv(os.path.join(args.out_dir, "tier3_5_gcomp_summary.csv"), index=False)

    print("\n" + "=" * 110)
    print("(A) ROUTER AUDIT — 4 cong thuc da khoa, baseline ngau nhien = 0.50")
    print("=" * 110)
    print(router_summary.to_string(index=False))

    print("\n" + "=" * 110)
    print("(B) STOCHASTIC CONTROL cho Complementarity Gain (doi chieu voi Tier 3)")
    print("=" * 110)
    print(gcomp_summary.to_string(index=False))
    print("\nSo sanh voi G_comp(raw,clahe) tai epoch 3 tu Tier 3:")
    print("  chase2rite: G_comp(raw,clahe) ~ 0.1312 (trung binh 2 seed)")
    print("  hrf2chase:  G_comp(raw,clahe) ~ 0.1153 (trung binh 2 seed)")

    print("\nDoc ket qua:")
    print("  (A) Neu ca 4 candidate ~0.50 -> Tier 3 ceiling that nhung KHONG the")
    print("      khai thac bang tin hieu unsupervised -> DUNG xay router, method")
    print("      that bai o day, khong phai o kien truc.")
    print("      Neu R_combo > cac candidate khac va > 0.5 ro ret o CA HAI shift")
    print("      -> co co so xay router that tu R_flip*R_temp.")
    print("  (B) Neu g_comp_stochastic (raw_A vs raw_B) XAP XI G_comp(raw,clahe)")
    print("      -> complementarity KHONG dac thu appearance, chi la hieu ung")
    print("      ensemble chung chung -> khong can CLAHE, 1 ensemble 2-seed thuong")
    print("      co the dat hieu qua tuong duong voi chi phi thap hon nhieu.")
    print("      Neu g_comp_stochastic NHO HON RO RET G_comp(raw,clahe)")
    print("      -> complementarity that su den tu appearance, dang xay kien truc.")


if __name__ == "__main__":
    main()
