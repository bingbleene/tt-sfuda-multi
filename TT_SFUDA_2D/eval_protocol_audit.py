"""
eval_protocol_audit.py

MUC DICH — audit 3-way, KHONG train gi, de xac dinh chinh xac nguyen nhan
HRF->CHASE raw (va cac shift khac) cho Dice khac nhau ~3 diem giua
validate() that cua repo va selective_fusion_eval.py.

GIA THUYET (theo dung phan tich da xac nhan): selective_fusion_eval.py
du doan o RESOLUTION MODEL (vd 384) roi UPSAMPLE prediction len NATIVE
resolution truoc khi so voi NATIVE GT — day la KHONG GIONG voi validate()
that cua repo, von danh gia HOAN TOAN o RESOLUTION MODEL (ca prediction
lan GT deu di qua Resize(config['input_h'], config['input_w']) trong
Dataset/Albumentations pipeline).

3 PHEP TINH SO SANH TREN CUNG 1 CHECKPOINT:
    A = validate() THAT cua repo (import truc tiep tu tt_sfuda_2d.py,
        khong viet lai) — day la "ground truth protocol" can khop theo.
    B = tu tay lap qua CUNG val_loader (Dataset/Albumentations that cua
        repo, GT da o RESOLUTION MODEL) — kiem tra ham dice_score cua
        minh cho ket qua tuong duong A (sanity check chinh minh).
    C = protocol CU cua selective_fusion_eval.py: du doan o resolution
        model, upsample prediction len native (cv2.INTER_LINEAR), so voi
        native GT doc truc tiep tu file — DAY LA PROTOCOL NGHI CO BUG.

Neu A ~ B va C lech ro so voi A/B -> xac nhan dung nguyen nhan la
evaluation-space mismatch, KHONG phai training regression.

CHAY (trong TT_SFUDA_2D/):
    python run.py eval_protocol_audit.py \
        --domain_shift hrf2chase --view raw --seed 1 \
        --ckpt_root /kaggle/working/outputs/full_checkpoints
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
import losses
from dataset import Dataset
from tt_sfuda_2d import validate
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from clahe_transform import ClaheLAB
from patch_inference import normalize_like_dataset

ALL_SHIFTS = {
    "chase2hrf": 1024,
    "chase2rite": 768,
    "hrf2chase": 384,
    "hrf2rite": 512,
}
THRESHOLD = 0.5


def load_config(domain_shift: str) -> dict:
    source, target = domain_shift.split("2")
    cfg_path = os.path.join("models", f"{source}_unet", f"config_{target}_dualema.yml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def dice_score(pred_bool: np.ndarray, gt_bool: np.ndarray) -> float:
    inter = int((pred_bool & gt_bool).sum())
    denom = int(pred_bool.sum()) + int(gt_bool.sum())
    return 1.0 if denom == 0 else 2.0 * inter / denom


def build_val_transform(cfg: dict, use_clahe: bool) -> Compose:
    """Y HET dong 295-299 cua tt_sfuda_2d_dualema.py — khong viet lai khac."""
    return Compose([
        *([ClaheLAB(clip_limit=2.0)] if use_clahe else []),
        Resize(cfg["input_h"], cfg["input_w"]),
        transforms.Normalize(),
    ])


def run_audit(domain_shift: str, view: str, seed: int, device: str, ckpt_root: str):
    cfg = load_config(domain_shift)
    source, target = domain_shift.split("2")
    use_clahe = (view == "clahe")

    # QUAN TRONG — sua bug: tt_sfuda_2d_dualema.py khi train GHI DE
    # cfg['input_h']/cfg['input_w'] bang --input_size TRUOC khi build
    # transform. Ban dau file nay quen lam buoc do nen doc thang YAML goc
    # (mac dinh 512 cho ca 4 shift) -> A/B bi danh gia SAI resolution
    # (512 thay vi 384/768/...). Phai override GIONG HET training script.
    resolution = ALL_SHIFTS[domain_shift]
    cfg["input_h"] = resolution
    cfg["input_w"] = resolution

    ckpt_path = os.path.join(ckpt_root, domain_shift, f"{view}_seed{seed}", "model.pth")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Thieu checkpoint: {ckpt_path}")

    model = archs.__dict__[cfg["arch"]](
        cfg["num_classes"], cfg["input_channels"], cfg["deep_supervision"]
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    criterion = losses.__dict__[cfg["loss"]]().to(device)

    val_img_ids = glob(os.path.join("inputs", target, "test", "images", "*" + cfg["img_ext"]))
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

    # ============================================================
    # A — validate() THAT cua repo, khong sua
    # ============================================================
    val_log = validate(val_loader, model, criterion)
    dice_A = val_log["dice"]

    # ============================================================
    # B — tu lap qua CUNG val_loader (GT da o resolution model qua
    #     chinh Dataset/Albumentations that), dung dice_score cua minh
    # ============================================================
    dice_B_list = []
    first_batch_debug_printed = False
    with torch.no_grad():
        for batch in val_loader:
            inp, tgt = batch[0], batch[1]
            if not first_batch_debug_printed:
                # KIEM TRA GIA DINH: mask co dung range [0,1] va dung thu tu
                # (input, target, ...) khong — neu gia dinh sai, dice_B se
                # sai am tham. In ra de tu xac nhan truoc khi tin dice_B.
                print(f"  [DEBUG batch0] inp.shape={tuple(inp.shape)} "
                      f"tgt.shape={tuple(tgt.shape)} "
                      f"tgt.dtype={tgt.dtype} "
                      f"tgt.min={tgt.min().item():.3f} tgt.max={tgt.max().item():.3f}")
                assert inp.shape[-1] == resolution and inp.shape[-2] == resolution, (
                    f"BUG: inp resolution {tuple(inp.shape[-2:])} != {resolution} "
                    f"da khoa cho {domain_shift} — kiem tra lai override cfg['input_h']/['input_w']."
                )
                first_batch_debug_printed = True
            inp = inp.to(device)
            out = model(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]
            prob = torch.sigmoid(out).cpu().numpy()
            pred_bool = (prob >= THRESHOLD)
            gt_bool = (tgt.numpy() >= 0.5)
            for b in range(pred_bool.shape[0]):
                dice_B_list.append(dice_score(pred_bool[b, 0], gt_bool[b, 0]))
    dice_B = float(np.mean(dice_B_list))

    # ============================================================
    # C — protocol CU cua selective_fusion_eval.py (nghi co bug):
    #     du doan o resolution model, upsample len native, so voi
    #     native GT doc truc tiep tu file
    # ============================================================
    img_dir = os.path.join("inputs", target, "test", "images")
    mask_dir = os.path.join("inputs", target, "test", "masks", "0")

    dice_C_list = []
    for iid in val_img_ids:
        img_path = os.path.join(img_dir, iid + cfg["img_ext"])
        mask_path = os.path.join(mask_dir, iid + cfg["mask_ext"])
        if not os.path.exists(mask_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]

        img_view = img.copy()
        if use_clahe:
            lab = cv2.cvtColor(img_view, cv2.COLOR_BGR2LAB)
            l, a, bb = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            img_view = cv2.cvtColor(cv2.merge([l_eq, a, bb]), cv2.COLOR_LAB2BGR)

        img_resized = cv2.resize(img_view, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        img_norm = normalize_like_dataset(img_resized)
        x = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            prob = torch.sigmoid(out).squeeze().cpu().numpy()

        prob_native = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_bool = prob_native >= THRESHOLD
        gt_bool = gt > 127
        dice_C_list.append(dice_score(pred_bool, gt_bool))
    dice_C = float(np.mean(dice_C_list)) if dice_C_list else float("nan")

    return {
        "domain_shift": domain_shift, "view": view, "seed": seed,
        "dice_A_repo_validate": dice_A,
        "dice_B_manual_model_res": dice_B,
        "dice_C_upsample_native": dice_C,
        "A_minus_C": dice_A - dice_C,
        "B_minus_C": dice_B - dice_C,
        "A_minus_B": dice_A - dice_B,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_shifts", nargs="+", default=["hrf2chase", "chase2rite"],
                         choices=list(ALL_SHIFTS.keys()))
    parser.add_argument("--views", nargs="+", default=["raw"], choices=["raw", "clahe"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_root", default="/kaggle/working/outputs/full_checkpoints")
    parser.add_argument("--out_dir", default="/kaggle/working/outputs/eval_audit")
    args = parser.parse_args()

    rows = []
    for shift in args.domain_shifts:
        for view in args.views:
            for seed in args.seeds:
                print(f"[{shift} {view} seed={seed}] dang audit...")
                row = run_audit(shift, view, seed, args.device, args.ckpt_root)
                rows.append(row)
                print(f"  A(repo validate)   = {row['dice_A_repo_validate']:.4f}")
                print(f"  B(manual, model-res)= {row['dice_B_manual_model_res']:.4f}")
                print(f"  C(upsample native)  = {row['dice_C_upsample_native']:.4f}")
                print(f"  A-C = {row['A_minus_C']:.4f}   B-C = {row['B_minus_C']:.4f}   A-B = {row['A_minus_B']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "eval_protocol_audit.csv")
    df.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 100)
    print("EVAL PROTOCOL AUDIT")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nLuu: {out_path}")
    print("\nDoc ket qua:")
    print("  - Neu A ~ B (chenh nho, vai phan nghin) VA |A-C| lon (~vai % diem Dice)")
    print("    -> XAC NHAN: khong co training regression, chi la evaluation-space")
    print("    mismatch. Can sua selective_fusion_eval.py dung protocol A/B.")
    print("  - Neu A khac B dang ke -> ham dice_score cua minh hoac cach lap batch")
    print("    co van de, can xem lai TRUOC KHI ket luan ve C.")


if __name__ == "__main__":
    main()
