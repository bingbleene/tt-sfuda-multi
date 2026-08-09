"""
validate_patch_inference.py
==============================
GIAI DOAN 1 cua y tuong "but pha" - KHONG TRAIN LAI GI, chi doi cach
INFERENCE tren CUNG 1 checkpoint da co san, de co bang chung nhanh nhat
truoc khi quyet dinh dau tu Giai doan 2 (train lai voi patch).

So sanh 2 cach tren CUNG anh, CUNG GT, CUNG model weights:
  (A) CU: resize ca anh ve input_h x input_w, predict, resize nguoc ket
      qua ve kich thuoc goc, so Dice voi GT native.
  (B) MOI: cat patch kich thuoc input_h x input_w TU ANH GOC (khong
      resize), predict tung patch, ghep lai o DUNG do phan giai native,
      so Dice voi GT native (khong qua buoc resize nao ca 2 chieu).

Cach dung:
    python run.py validate_patch_inference.py --source chase_unet \\
        --target hrf --checkpoint cache/stage1_chase_hrf.pth --n_images 5
"""
import os
import argparse
from glob import glob

import cv2
import yaml
import numpy as np
import torch

import archs
from patch_inference import patch_predict_native, normalize_like_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n_images', type=int, default=5)
    parser.add_argument('--overlap_ratio', type=float, default=0.5)
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

    patch_size = (config['input_h'], config['input_w'])

    dice_resize_list, dice_patch_list = [], []

    for img_path in img_paths:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, img_id + config['mask_ext'])
        if not os.path.exists(mask_path):
            print(f"  [BO QUA] khong tim thay mask cho {img_id}")
            continue

        # GIU NGUYEN BGR - dataset.py that KHONG chuyen RGB (cv2.imread mac dinh la BGR).
        img_bgr = cv2.imread(img_path)
        native_h, native_w = img_bgr.shape[:2]

        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_native = cv2.resize(gt, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
        gt_binary = (gt_native > 127).astype(np.uint8)

        # --- (A) CU: resize toan anh ---
        img_resized = cv2.resize(img_bgr, (config['input_w'], config['input_h']))
        img_norm = normalize_like_dataset(img_resized)  # khop dung dataset.py (Normalize + /255 lan 2)
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
        with torch.no_grad():
            prob_small = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]
        prob_resize_native = cv2.resize(prob_small, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
        pred_resize = (prob_resize_native > args.threshold).astype(np.uint8)
        d_resize = dice_score(pred_resize, gt_binary)

        # --- (B) MOI: patch tu anh goc, khong resize ---
        prob_patch_native = patch_predict_native(model, img_bgr, patch_size,
                                                   overlap_ratio=args.overlap_ratio)
        pred_patch = (prob_patch_native > args.threshold).astype(np.uint8)
        d_patch = dice_score(pred_patch, gt_binary)

        dice_resize_list.append(d_resize)
        dice_patch_list.append(d_patch)
        print(f"  {img_id} ({native_h}x{native_w}): Dice resize-toan-anh={d_resize:.4f}, "
              f"Dice patch-native={d_patch:.4f}, chenh={d_patch - d_resize:+.4f}")

    mean_resize = np.mean(dice_resize_list)
    mean_patch = np.mean(dice_patch_list)

    print("\n" + "=" * 75)
    print(f"Dice TRUNG BINH (A) resize-toan-anh (cach cu): {mean_resize:.4f}")
    print(f"Dice TRUNG BINH (B) patch-native (cach moi):   {mean_patch:.4f}  "
          f"({'+' if mean_patch >= mean_resize else ''}{mean_patch - mean_resize:.4f})")
    if mean_patch > mean_resize + 0.02:
        print("-> TIN HIEU MANH: patch-inference tot hon RO RANG du CHUA train lai gi.")
        print("   Buoc tiep theo: viet pipeline train Stage I/II tren DU LIEU PATCH (Giai doan 2)")
        print("   - se con tot hon nua vi model se duoc HOC truc tiep tren du lieu day du chi tiet,")
        print("   khong chi infer o do phan giai cao voi trong so hoc tu du lieu bi mat thong tin.")
    elif mean_patch > mean_resize:
        print("-> Tin hieu duong nhung nho. Co the can Giai doan 2 (train lai tren patch) moi thay")
        print("   ro ret qua, vi trong so hien tai van HOC tu du lieu da resize/mat thong tin.")
    else:
        print("-> Khong cai thien (hoac te hon) du gia thuyet resolution co ve dung theo so lieu")
        print("   check_resolution_gap.py. Co the do domain-shift o thong ke mau sac/tuong phan")
        print("   tung patch khac voi toan anh (patch chi thay 1 vung nho, mat ngu canh toan cuc)")
        print("   at che mat loi ich tu do phan giai cao hon - can xem xet ky truoc khi ket luan.")
    print("=" * 75)


if __name__ == '__main__':
    main()
