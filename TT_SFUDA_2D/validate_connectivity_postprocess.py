"""
validate_connectivity_postprocess.py
=======================================
Validate largest_component_prior() TRUC TIEP tren 1 checkpoint DA CO SAN
(khong can train gi moi) - do Dice truoc/sau khi loc lien thong tren tap
VAL CO GT THAT (khac Frangi/wavelet/SAM - o day khong can so voi mach mau
"tu tim ra tu dau", chi can so 2 phien ban cua CUNG 1 prediction).

Cach dung (dung checkpoint stage1 cache DA CO, hoac bat ky model.pth nao):
    python run.py validate_connectivity_postprocess.py --source chase_unet \\
        --target hrf --checkpoint cache/stage1_chase_hrf.pth
"""
import os
import argparse

import cv2
import yaml
import numpy as np
import torch
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from glob import glob

import archs
from dataset import Dataset
from connectivity_prior import largest_component_prior


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--checkpoint', required=True,
                         help='Bat ky checkpoint UNet chuan nao (stage1_ckpt, model.pth nguon, '
                              'hoac checkpoint Stage II da luu) - script nay CHI danh gia hau xu ly, '
                              'khong quan tam checkpoint tu dau ra.')
    parser.add_argument('--keep_top_k', type=int, default=1)
    parser.add_argument('--min_component_size', type=int, default=20)
    parser.add_argument('--dilate_before_merge', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()


def dice_score(pred_binary, gt_binary):
    pred_binary = pred_binary.astype(np.uint8)
    gt_binary = gt_binary.astype(np.uint8)
    inter = (pred_binary & gt_binary).sum()
    denom = pred_binary.sum() + gt_binary.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def main():
    args = parse_args()

    config_file = "config_" + args.target + "_dualema"
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    val_img_ids = glob(os.path.join('inputs', args.target, 'test', 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']), transforms.Normalize(),
    ])
    val_dataset = Dataset(
        img_ids=val_img_ids, img_dir=os.path.join('inputs', args.target, 'test', 'images'),
        mask_dir=os.path.join('inputs', args.target, 'test', 'masks'),
        img_ext=config['img_ext'], mask_ext=config['mask_ext'],
        num_classes=config['num_classes'], transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'],
                                            config['deep_supervision']).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    dice_before_list, dice_after_list = [], []
    n_recovered, n_hurt, n_unchanged = 0, 0, 0

    with torch.no_grad():
        for input, target, meta in val_loader:
            input = input.cuda()
            output = model(input)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            gt = target.numpy()[0, 0]

            pred_before = (prob > args.threshold).astype(np.uint8)
            pred_after = largest_component_prior(
                pred_before, keep_top_k=args.keep_top_k,
                min_component_size=args.min_component_size,
                dilate_before_merge=args.dilate_before_merge)

            d_before = dice_score(pred_before, gt)
            d_after = dice_score(pred_after, gt)
            dice_before_list.append(d_before)
            dice_after_list.append(d_after)

            diff = d_after - d_before
            if diff > 1e-4:
                n_recovered += 1
            elif diff < -1e-4:
                n_hurt += 1
            else:
                n_unchanged += 1

            img_id = meta['img_id'][0] if isinstance(meta, dict) else meta
            print(f"  {img_id}: Dice truoc={d_before:.4f}, sau={d_after:.4f}, "
                  f"chenh={diff:+.4f}")

    mean_before = np.mean(dice_before_list)
    mean_after = np.mean(dice_after_list)

    print("\n" + "=" * 70)
    print(f"Dice TRUNG BINH truoc loc lien thong: {mean_before:.4f}")
    print(f"Dice TRUNG BINH SAU loc lien thong:   {mean_after:.4f}  "
          f"({'+' if mean_after >= mean_before else ''}{mean_after - mean_before:.4f})")
    print(f"So anh: cai thien={n_recovered}, xau di={n_hurt}, khong doi={n_unchanged} "
          f"(tren {len(dice_before_list)} anh)")
    if mean_after > mean_before and n_hurt <= n_recovered:
        print("-> TICH CUC: dang tich hop vao pipeline chinh nhu 1 buoc hau xu ly "
              "SAU khi model du doan (khong can train lai, ap dung ngay cho ca 4 domain shift).")
    elif n_hurt > n_recovered:
        print("-> CANH BAO: co nhieu anh bi loc lien thong lam XAU DI hon la cai thien - "
              "co the domain shift lam mach mau THAT bi dut nhieu, khien component nho lai "
              "la phan dung. Thu tang --dilate_before_merge hoac --keep_top_k truoc khi ket luan.")
    else:
        print("-> Khong ro rang - can xem chi tiet tung anh bi 'xau di' de hieu nguyen nhan.")
    print("=" * 70)


if __name__ == '__main__':
    main()
