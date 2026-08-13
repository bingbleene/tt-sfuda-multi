"""
tier2_survival.py

MUC DICH (Tier 2 — LAN DAU CAN TRAIN THAT, van khong doc nhan)
------------------------------------------------------------------
Tier 0 + Tier 1 da xong: raw/CLAHE tao disagreement lon (D_fg_union~0.45-0.58)
va disagreement do co cau truc (Frangi enrichment phan hoa dung huong voi
lich su that: C->R tot nhat, H->C xau nhat). Nhung ca hai deu do tren SOURCE
MODEL DONG BANG — khong chung minh duoc divergence song sot qua qua trinh
Stage II adaptation (dung cau bay da giet Region/Topology dual-pair truoc
day: EMA lag khien 2 nhanh hoi tu ve nhau du xuat phat khac nhau that).

Tier 2 tra loi dung cau hoi do bang thiet ke KHONG THE thay the:
    4 model / shift, CHUNG 1 Stage-I checkpoint, chi khac (a) raw/CLAHE
    view trong Stage II va (b) seed A/B (stochastic training dynamics):

        Raw_A    (seed A, raw view)
        Raw_B    (seed B, raw view)
        CLAHE_A  (seed A, CLAHE view)
        CLAHE_B  (seed B, CLAHE view)

    D_stoch(e) = mean( D(Raw_A,Raw_B,e), D(CLAHE_A,CLAHE_B,e) )   <- stochastic control
    D_app(e)   = mean( D(Raw_A,CLAHE_A,e), D(Raw_B,CLAHE_B,e) )   <- appearance treatment,
                                                                       matched seed
    delta_D(e) = D_app(e) - D_stoch(e)
    S_D        = delta_D(3) / (delta_D(0) + eps)      <- % divergence song sot

Tinh RIENG cho D_fg_union, D_prob_fg, D_skeleton (khong dung D_all, khong
dung clskeleton_overlap/fg_ratio_change o Tier nay — do la cong cu cua
Tier 0/1, khong phai cau hoi Tier 2 dang hoi).

PILOT — 2 shift cuc doan nhat theo Tier 1, falsification power cao nhat:
    chase2rite  (768px)  — case TOT nhat  (recovered_vessel_like=0.697,
                                            background_explosion=0.223)
    hrf2chase   (384px)  — case XAU nhat  (recovered_vessel_like=0.373,
                                            background_explosion=0.568)

RANH GIOI BAT BUOC
--------------------
- KHONG doc target mask/label o bat ky dong nao trong file nay.
- rollback = OFF, early_stop_unsupervised = OFF (khong truyen 2 flag nay
  cho tt_sfuda_2d_dualema.py). Da xac nhan tu code that: khong co CLI
  --rollback rieng, rollback chi xay ra qua early_stopper.restore_best()
  khi --early_stop_unsupervised duoc bat — nen chi can KHONG bat co do.
- Da xac nhan tu code that: --use_clahe ANH HUONG Stage I (ClaheLAB nam
  trong train_transform dung ca 2 stage). NHUNG vi Tier 2 SKIP Stage I
  bang cach truyen --stage1_ckpt co san (msrc_model.load_state_dict(...)
  bo qua toan bo vong for epoch in range(config['stage1'])), CLAHE
  KHONG con anh huong gi o buoc nay — ca 4 model (Raw_A/B, CLAHE_A/B)
  dung CHUNG 1 checkpoint Stage-I duoc train bang RAW (khong CLAHE).
  --use_clahe chi bat dau co tac dung khi Stage II doc train_loader.
  => Day chinh xac la "Stage-II-only appearance treatment" ma Tier 2
  muon do.
- Da xac nhan tu code that: train_loader (co mask) van duoc dung lam
  stage2_train_loader khi khong bat early-stop, VA validate() doc
  target test mask sau moi epoch (val_log/epoch_val_log) — day la
  MASK LEAKAGE thuc su neu chay nguyen ban CLI hien tai, DU mask khong
  vao backward. Vi vay tier2_survival.py KHONG duoc goi
  tt_sfuda_2d_dualema.py nguyen ban — can them --tier2_diagnostic vao
  chinh file training (bo val_loader, bo doc mask, luu checkpoint moi
  epoch, cudnn.deterministic=True). Xem TODO trong run_stage2_training().
- Khong co --resolution CLI — resolution lay tu config['input_h']/
  ['input_w'] qua Resize(...). Can them --input_size override (hoac
  sua truc tiep input_h/input_w trong config_{target}_dualema.yml truoc
  khi chay, nhung --input_size sach hon, khong dung cham file dung cho
  cac thi nghiem khac).
- Checkpoint PHAI luu tai e=0,1,2,3 trong CUNG MOT lan train 3 epoch — KHONG
  chay stage2_epochs=1 roi 2 roi 3 rieng le (RNG/dataloader prefix co the
  khong giong nhau giua cac lan chay doc lap).

TRUOC KHI CHAY — CAC INTEGRATION POINT BAT BUOC PHAI NOI:
    1. tt_sfuda_2d_dualema.py CAN THEM (o chinh file training, chua lam
       o day vi can xem toan bo file that truoc khi vá an toan):
           --tier2_diagnostic   (bool) — tat val_loader, tat doc mask,
                                 luu tgt_model.state_dict() tai epoch_0
                                 (ngay sau khi load stage1_ckpt) va
                                 epoch_1..N (cuoi moi epoch Stage II),
                                 ep cudnn.benchmark=False,
                                 cudnn.deterministic=True
           --tier2_ckpt_dir     (str) — thu muc luu epoch_*.pth
           --input_size         (int) — override input_h/input_w tu CLI
    2. run_stage2_training()   — goi training that qua subprocess, tra ve
       dict {epoch: ckpt_path}, SAU KHI (1) da lam xong
    3. STAGE1_CKPT              — duong dan Stage-I checkpoint da cache
       (1 checkpoint/shift, train bang RAW, dung chung cho ca 4 model)
    4. load_arch()              — xay model tu cfg (archs.__dict__[...])
"""

import os
import argparse
from glob import glob

import cv2
import numpy as np
import pandas as pd
import yaml
import torch
from skimage.morphology import skeletonize

import archs
from patch_inference import normalize_like_dataset

# ============================================================
# CONFIG
# ============================================================

PILOT_SHIFTS = {
    "chase2rite": 768,   # case TOT nhat theo Tier 1
    "hrf2chase": 384,    # case XAU nhat theo Tier 1
}

SEEDS = {"A": 1001, "B": 2002}   # PHAI khac nhau, giu co dinh xuyen suot pilot
EPOCHS = [0, 1, 2, 3]
THRESHOLD = 0.5
EPS = 1e-7

# Da xac nhan: --use_clahe anh huong Stage I trong code goc, NHUNG Tier 2
# luon SKIP Stage I bang --stage1_ckpt co san (xem docstring dau file) —
# nen ca 4 model dung CHUNG 1 checkpoint duoi day, train bang RAW.
# Khong con nhanh re theo CLAHE_APPLIES_TO_STAGE1 nua.
STAGE1_CKPT = {
    "chase2rite": "cache/chase_to_rite_stage1_raw.pth",
    "hrf2chase": "cache/hrf_to_chase_stage1_raw.pth",
}


# ============================================================
# INTEGRATION POINTS — noi voi training pipeline that truoc khi chay
# ============================================================

def run_stage2_training(
    domain_shift: str,
    resolution: int,
    appearance_view: str,   # "raw" hoac "clahe"
    seed: int,
    stage1_ckpt: str,
    out_dir: str,
) -> dict:
    """
    Da noi voi tt_sfuda_2d_dualema.py ban vá (them --tier2_diagnostic /
    --tier2_ckpt_dir / --input_size). Khong dung --topology/--slow_keep_rate
    rieng — Tier 2 giu nguyen default cua config (topology='parallel',
    ensemble_mode='mean'), vi cau hoi Tier 2 la "dung co che Stage II THAT
    (Dual-EMA) co giu duoc diversity khong", khong phai thu bien the khac.
    """
    import subprocess

    source_domain, target = domain_shift.split("2")
    source = f"{source_domain}_unet"

    cmd = [
        "python", "run.py", "tt_sfuda_2d_dualema.py",
        "--source", source,
        "--target", target,
        "--input_size", str(resolution),
        "--stage1_ckpt", stage1_ckpt,
        "--stage2_epochs", "3",
        "--seed", str(seed),
        "--tier2_diagnostic",
        "--tier2_ckpt_dir", out_dir,
    ]
    if appearance_view == "clahe":
        cmd.append("--use_clahe")
    # KHONG truyen --early_stop_unsupervised (=> rollback cung OFF, vi
    # rollback chi xay ra qua early_stopper.restore_best()).

    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    ckpts = {e: os.path.join(out_dir, f"epoch_{e}.pth") for e in EPOCHS}
    for e, path in ckpts.items():
        if not os.path.exists(path):
            raise RuntimeError(f"Thieu checkpoint epoch {e}: {path} (training co the da loi).")
    return ckpts


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


# ============================================================
# CLAHE + preprocess — dung nguyen ban da xac nhan (Tier 0/1)
# ============================================================

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


def get_diagnostic_image_paths(target: str, cfg: dict):
    """Giu dung tap anh train/images da dung o Tier 0/1, de so sanh xuyen suot."""
    img_dir = os.path.join("inputs", target, "train", "images")
    paths = sorted(glob(os.path.join(img_dir, "*" + cfg["img_ext"])))
    if not paths:
        raise RuntimeError(f"Khong tim thay anh: {img_dir}")
    return paths


# ============================================================
# Pairwise disagreement — D_fg_union, D_prob_fg, D_skeleton
# (cong thuc giong check_raw_clahe_disagreement.py, nhung giua 2 PROB MAP
# tu 2 MODEL khac nhau, khong phai 2 view cua cung 1 model)
# ============================================================

def pairwise_disagreement(prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    mask_a = prob_a >= THRESHOLD
    mask_b = prob_b >= THRESHOLD
    fg_union = mask_a | mask_b
    n_fg = int(fg_union.sum())
    hard_diff = mask_a != mask_b

    if n_fg > 0:
        d_fg = float(hard_diff[fg_union].sum() / n_fg)
        d_prob = float(np.abs(prob_a[fg_union] - prob_b[fg_union]).mean())
    else:
        d_fg = 0.0
        d_prob = 0.0

    skel = skeletonize(fg_union)
    n_skel = int(skel.sum())
    d_skel = float(hard_diff[skel].sum() / n_skel) if n_skel > 0 else 0.0

    return {"D_fg_union": d_fg, "D_prob_fg": d_prob, "D_skeleton": d_skel}


# ============================================================
# Main — 1 shift
# ============================================================

def run_pilot_shift(domain_shift: str, resolution: int, device: str, out_dir: str, max_images):
    stage1_ckpt = STAGE1_CKPT[domain_shift]
    if stage1_ckpt is None:
        raise RuntimeError(f"STAGE1_CKPT['{domain_shift}'] chua duoc dien.")

    cfg = load_config(domain_shift)
    _, target = domain_shift.split("2")
    paths = get_diagnostic_image_paths(target, cfg)
    if max_images is not None:
        paths = paths[:max_images]
    raw_images = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
    clahe_images = [apply_clahe_lab(im) for im in raw_images]

    # --- Train 4 model, lay checkpoint tai e=0,1,2,3 trong cung 1 run ---
    ckpts = {}
    for view in ["raw", "clahe"]:
        for seed_name, seed_val in SEEDS.items():
            key = f"{view}_{seed_name}"
            run_out = os.path.join(out_dir, domain_shift, key)
            print(f"[TRAIN] {domain_shift} {key} (seed={seed_val}, view={view})")
            ckpts[key] = run_stage2_training(
                domain_shift, resolution, view, seed_val, stage1_ckpt, run_out
            )

    # --- Voi tung epoch, load 4 model, do 4 cap disagreement ---
    rows = []
    for e in EPOCHS:
        models = {}
        for view in ["raw", "clahe"]:
            for seed_name in SEEDS:
                key = f"{view}_{seed_name}"
                model = load_arch(cfg, device)
                load_checkpoint_into(model, ckpts[key][e], device)
                models[key] = model

        per_image = {"RR": [], "CC": [], "RA_CA": [], "RB_CB": []}
        for raw_img, clh_img in zip(raw_images, clahe_images):
            p_raw_a = predict_prob(models["raw_A"], raw_img, resolution, device)
            p_raw_b = predict_prob(models["raw_B"], raw_img, resolution, device)
            p_clh_a = predict_prob(models["clahe_A"], clh_img, resolution, device)
            p_clh_b = predict_prob(models["clahe_B"], clh_img, resolution, device)

            per_image["RR"].append(pairwise_disagreement(p_raw_a, p_raw_b))
            per_image["CC"].append(pairwise_disagreement(p_clh_a, p_clh_b))
            per_image["RA_CA"].append(pairwise_disagreement(p_raw_a, p_clh_a))
            per_image["RB_CB"].append(pairwise_disagreement(p_raw_b, p_clh_b))

        def avg(key, metric):
            return float(np.mean([r[metric] for r in per_image[key]]))

        row = {"domain_shift": domain_shift, "epoch": e}
        for metric in ["D_fg_union", "D_prob_fg", "D_skeleton"]:
            d_rr = avg("RR", metric)
            d_cc = avg("CC", metric)
            d_ra_ca = avg("RA_CA", metric)
            d_rb_cb = avg("RB_CB", metric)
            d_stoch = (d_rr + d_cc) / 2.0
            d_app = (d_ra_ca + d_rb_cb) / 2.0
            suffix = metric.replace("D_", "")
            row[f"D_raw_raw_{suffix}"] = d_rr
            row[f"D_clahe_clahe_{suffix}"] = d_cc
            row[f"D_raw_clahe_seedA_{suffix}"] = d_ra_ca
            row[f"D_raw_clahe_seedB_{suffix}"] = d_rb_cb
            row[f"D_app_{suffix}"] = d_app
            row[f"D_stoch_{suffix}"] = d_stoch
            row[f"delta_D_{suffix}"] = d_app - d_stoch
        rows.append(row)
        print(f"[{domain_shift}] epoch {e} done")

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"tier2_survival_{domain_shift}.csv")
    df.to_csv(csv_path, index=False)
    return df


def gate_check(df: pd.DataFrame, domain_shift: str) -> dict:
    row3 = df[df["epoch"] == 3].iloc[0]
    row0 = df[df["epoch"] == 0].iloc[0]

    pass_fg = row3["delta_D_fg_union"] > 0
    pass_skel = row3["delta_D_skeleton"] > 0
    matched_seed_fg = (row3["D_raw_clahe_seedA_fg_union"] > row3["D_raw_raw_fg_union"]) and \
                       (row3["D_raw_clahe_seedB_fg_union"] > row3["D_clahe_clahe_fg_union"])

    eps = EPS
    s_d_fg = row3["delta_D_fg_union"] / (row0["delta_D_fg_union"] + eps)
    s_d_skel = row3["delta_D_skeleton"] / (row0["delta_D_skeleton"] + eps)

    collapse_warning = (row0["delta_D_fg_union"] > 0.05) and (row3["delta_D_fg_union"] < 0.01)

    return {
        "domain_shift": domain_shift,
        "gate_pass_fg": bool(pass_fg),
        "gate_pass_skeleton": bool(pass_skel),
        "matched_seed_consistent_fg": bool(matched_seed_fg),
        "survival_ratio_fg": float(s_d_fg),
        "survival_ratio_skeleton": float(s_d_skel),
        "collapse_warning": bool(collapse_warning),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_shifts", nargs="+", default=list(PILOT_SHIFTS.keys()),
                         choices=list(PILOT_SHIFTS.keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/tier2_survival")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    all_gates = []
    for shift in args.domain_shifts:
        resolution = PILOT_SHIFTS[shift]
        df = run_pilot_shift(shift, resolution, args.device, args.out_dir, args.max_images)
        gate = gate_check(df, shift)
        all_gates.append(gate)

        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", None)
        print(f"\n=== {shift} — trajectory (fg_union / skeleton) ===")
        print(df[[
            "epoch",
            "D_raw_raw_fg_union", "D_clahe_clahe_fg_union",
            "D_raw_clahe_seedA_fg_union", "D_raw_clahe_seedB_fg_union",
            "D_app_fg_union", "D_stoch_fg_union", "delta_D_fg_union",
            "delta_D_skeleton",
        ]].to_string(index=False))

    gates_df = pd.DataFrame(all_gates)
    gates_path = os.path.join(args.out_dir, "tier2_gate_summary.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    gates_df.to_csv(gates_path, index=False)

    print("\n" + "=" * 100)
    print("TIER 2 GATE SUMMARY")
    print("=" * 100)
    print(gates_df.to_string(index=False))
    print(f"\nChi tiet: {gates_path}")
    print("\nDieu kien PASS (ca hai, khong chi trung binh):")
    print("  delta_D_fg_union(e=3) > 0  VA  delta_D_skeleton(e=3) > 0")
    print("Canh bao collapse: delta_D(0) lon nhung delta_D(3) gan 0")
    print("  -> divergence CHET trong training, giong dung that bai Region/Topology.")
    print("Thu hang C->R vs H->C la secondary evidence, KHONG phai hard gate —")
    print("Tier 1 do quality/type cua recovery, Tier 2 do su song sot cua diversity,")
    print("hai dai luong khong bat buoc cung ordering.")


if __name__ == "__main__":
    main()
