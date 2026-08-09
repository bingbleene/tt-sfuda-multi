"""
validate_clahe_prior.py
==========================
Y TUONG MOI (khac han resolution): pipeline UNet CHINH (dataset.py) hien
KHONG BAO GIO ap dung CLAHE len anh dua vao model - chi frangi_prior.py
dung CLAHE cho tinh toan RIENG cua no (Frangi), hoan toan tach biet voi
luong du lieu that su nuoi UNet.

CO SO: CLAHE (clip=2, grid=8x8) la buoc tien xu ly CHUAN trong hau het
benchmark vessel segmentation nghiem tuc (giam lech tuong phan/anh sang
giua cac he thong camera khac nhau). HRF va RITE co the den tu camera/
quy trinh chup rat khac nhau - day la loai "domain gap" REGISTRATION-
LEVEL (mau sac/tuong phan), khac han van de resolution (KHONG GIAN).

Literature (Regularizing Self-training via Structural Constraints,
arXiv:2305.00131) chi ro: self-training (Stage II) KHUYCH DAI loi tu tin
sai do tuong quan gia ve mau sac/anh sang hoc tu domain nguon - dung
CLAHE truoc de giam bot su khac biet nay CO THE giam confirmation bias,
dac biet quan trong voi domain gap lon (HRF->RITE, source-only chi 41%).

Script nay CHI DO (zero-shot, khong train) - so Dice CO/KHONG CLAHE tren
CUNG 1 checkpoint, giong dung tinh than da lam voi resolution scaling.

Cach dung:
    python run.py validate_clahe_prior.py --source hrf_unet --target rite \\
        --checkpoint cache/stage1_hrf_rite.pth --n_images 5
"""
import os
import argparse
from glob import glob

import cv2
import yaml
import numpy as np
import torch

import archs
from patch_inference import normalize_like_dataset


def apply_clahe_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Ap dung CLAHE DUNG CACH cho anh mau: chuyen sang LAB, CHI ap dung
    len kenh L (do sang) - GIU NGUYEN thong tin mau sac (kenh a,b), tranh
    bi bien dang mau nhu khi ap CLAHE truc tiep len tung kenh BGR rieng le."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n_images', type=int, default=5)
    parser.add_argument('--clip_limit', type=float, default=2.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def dice_score(pred_binary, gt_binary):
    pred_binary, gt_binary = pred_binary.astype(np.uint8), gt_binary.astype(np.uint8)
    inter = (pred_binary & gt_binary).sum()
    denom = pred_binary.sum() + gt_binary.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def main():
    args = parse_args()

    config_file = "config_" + args.target + "_dualema"
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    img_dir = os.path.join('inputs', args.target, 'test', 'images')
    mask_dir = os.path.join('inputs', args.target, 'test', 'masks', '0')
    img_paths = sorted(glob(os.path.join(img_dir, '*' + config['img_ext'])))
    rng = np.random.RandomState(args.seed)
    chosen = rng.choice(len(img_paths), size=min(args.n_images, len(img_paths)), replace=False)
    img_paths = [img_paths[i] for i in chosen]

    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'],
                                            config['deep_supervision']).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    dice_no_clahe_list, dice_clahe_list = [], []

    for img_path in img_paths:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, img_id + config['mask_ext'])
        if not os.path.exists(mask_path):
            print(f"  [BO QUA] khong tim thay mask cho {img_id}")
            continue

        img_bgr = cv2.imread(img_path)
        native_h, native_w = img_bgr.shape[:2]
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_native = cv2.resize(gt, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
        gt_binary = (gt_native > 127).astype(np.uint8)

        def predict(img_input_bgr):
            img_resized = cv2.resize(img_input_bgr, (config['input_w'], config['input_h']))
            img_norm = normalize_like_dataset(img_resized)
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
            with torch.no_grad():
                prob_small = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]
            prob_native = cv2.resize(prob_small, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
            return (prob_native > args.threshold).astype(np.uint8)

        pred_no_clahe = predict(img_bgr)
        img_clahe = apply_clahe_bgr(img_bgr, clip_limit=args.clip_limit)
        pred_clahe = predict(img_clahe)

        d_no_clahe = dice_score(pred_no_clahe, gt_binary)
        d_clahe = dice_score(pred_clahe, gt_binary)
        dice_no_clahe_list.append(d_no_clahe)
        dice_clahe_list.append(d_clahe)

        print(f"  {img_id}: Dice khong-CLAHE={d_no_clahe:.4f}, Dice co-CLAHE={d_clahe:.4f}, "
              f"chenh={d_clahe - d_no_clahe:+.4f}")

    mean_no_clahe = np.mean(dice_no_clahe_list)
    mean_clahe = np.mean(dice_clahe_list)

    print("\n" + "=" * 70)
    print(f"Dice TRUNG BINH khong CLAHE: {mean_no_clahe:.4f}")
    print(f"Dice TRUNG BINH co CLAHE:    {mean_clahe:.4f}  "
          f"({'+' if mean_clahe >= mean_no_clahe else ''}{mean_clahe - mean_no_clahe:.4f})")
    if mean_clahe > mean_no_clahe + 0.01:
        print("-> TIN HIEU DUONG: CLAHE giup ngay ca zero-shot. Dang thu train that "
              "(them CLAHE vao dataset.py that su, khong chi zero-shot).")
    elif mean_clahe > mean_no_clahe:
        print("-> Tin hieu duong nho, chua chac chan.")
    else:
        print("-> Khong cai thien. Domain gap co the KHONG chu yeu do mau sac/tuong "
              "phan - can xem xet nguyen nhan khac.")
    print("=" * 70)


if __name__ == '__main__':
    main()
