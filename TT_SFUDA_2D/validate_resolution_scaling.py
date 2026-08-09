"""
validate_resolution_scaling.py
=================================
SUA LOI THIET KE THI NGHIEM cua validate_patch_inference.py: patch-native
doi CUNG LUC 2 bien - (1) do phan giai VA (2) ty le vat the + ngu canh
toan cuc (1 patch 512x512 cat tu anh HRF 3504px chi thay ~15% chieu rong
mat, mach mau trong do trong TO HON RAT NHIEU LAN so voi trong anh da
resize toan cuc ma model quen nhin - UNet KHONG bat bien ty le neu khong
duoc train da ty le, nen co the khong nhan ra "vat the to bat thuong" do
la mach mau). Ket qua am cua patch KHONG bac bo gia thuyet resolution -
no chi cho biet model KHONG generalize zero-shot qua 1 thay doi ty le +
mat ngu canh dot ngot, la 1 cau hoi KHAC voi cau hoi ta muon hoi.

O DAY: chi tang kich thuoc RESIZE CUA CA ANH (512 -> 768 -> 1024 -> ...),
GIU NGUYEN toan bo anh trong khung nhin (ngu canh toan cuc + ty le vat
the tuong doi KHONG doi, chi so pixel bieu dien tang len). Day la phep do
CO LAP DUNG 1 bien (do phan giai), khong con nhieu do ty le/ngu canh.

UNet 5-level (4 lan pool/up) can kich thuoc chia het cho 16 de khop chieu
khi concat skip connection - cac gia tri thu duoi day (512/768/1024/1280)
deu chia het cho 16, an toan.

Cach dung:
    python run.py validate_resolution_scaling.py --source chase_unet \\
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
from patch_inference import normalize_like_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n_images', type=int, default=5)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[512, 768, 1024, 1280],
                         help='Danh sach kich thuoc resize (vuong) can thu, PHAI chia het cho 16.')
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
    for r in args.resolutions:
        assert r % 16 == 0, f"Resolution {r} khong chia het cho 16 - UNet 5-level se loi khop chieu skip connection."

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

    # dice_by_res[resolution] = list Dice tren tung anh
    dice_by_res = {r: [] for r in args.resolutions}
    per_image_data = []

    for img_path in img_paths:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, img_id + config['mask_ext'])
        if not os.path.exists(mask_path):
            print(f"  [BO QUA] khong tim thay mask cho {img_id}")
            continue

        img_bgr = cv2.imread(img_path)  # GIU BGR - khop dataset.py that
        native_h, native_w = img_bgr.shape[:2]

        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_native = cv2.resize(gt, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
        gt_binary = (gt_native > 127).astype(np.uint8)

        row = {'img_id': img_id}
        for res in args.resolutions:
            img_resized = cv2.resize(img_bgr, (res, res))
            img_norm = normalize_like_dataset(img_resized)
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
            with torch.no_grad():
                prob_small = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]
            prob_native = cv2.resize(prob_small, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
            pred = (prob_native > args.threshold).astype(np.uint8)
            d = dice_score(pred, gt_binary)
            dice_by_res[res].append(d)
            row[res] = d

        per_image_data.append(row)
        row_str = ", ".join(f"{res}px={row[res]:.4f}" for res in args.resolutions)
        print(f"  {img_id} ({native_h}x{native_w}): {row_str}")

    print("\n" + "=" * 75)
    print(f"{'Resolution':<12} {'Dice trung binh':<18} {'So voi 512px'}")
    print("-" * 75)
    base_mean = np.mean(dice_by_res[args.resolutions[0]])
    for res in args.resolutions:
        mean_d = np.mean(dice_by_res[res])
        diff = mean_d - base_mean
        marker = "" if res == args.resolutions[0] else f"({'+' if diff >= 0 else ''}{diff:.4f})"
        print(f"{res}px{'':<7} {mean_d:.4f}{'':<13} {marker}")

    best_res = max(args.resolutions, key=lambda r: np.mean(dice_by_res[r]))
    best_mean = np.mean(dice_by_res[best_res])
    print("=" * 75)
    if best_res != args.resolutions[0] and best_mean > base_mean + 0.01:
        print(f"-> XU HUONG TANG THEO DO PHAN GIAI: {best_res}px tot nhat ({best_mean:.4f}, "
              f"+{best_mean - base_mean:.4f} so voi 512px). Day la bang chung SACH (khong con "
              f"nhieu ty le/ngu canh nhu patch) ung ho gia thuyet resolution. Buoc tiep theo: "
              f"XEM XET train lai (it nhat fine-tune ngan) o do phan giai {best_res}px thay vi "
              f"512px, roi so Dice that voi baseline 58.25/58.20.")
    elif best_mean <= base_mean + 0.01:
        print("-> KHONG co xu huong tang ro ret theo resolution (zero-shot). Co the checkpoint "
              "nay (chi Stage I, chua qua Stage II day du) chua phai dai dien tot nhat de test - "
              "hoac gia thuyet resolution that su khong phai nut that chinh. Can doi chieu voi "
              "checkpoint Stage II day du (58.20) truoc khi ket luan han.")
    print("=" * 75)


if __name__ == '__main__':
    main()
