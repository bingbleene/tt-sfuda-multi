"""
diagnose_gap_v2.py
====================
Round 2 chan doan domain gap. Gia thuyet: khi gap qua lon, Theta_s du doan
"toan nen" mot cach tu tin (entropy thap gia tao, nhu vua do o v1). Do do,
thay vi do entropy, do TY LE PIXEL DU DOAN DUONG TINH (mach mau) tren anh
DICH, roi so voi ty le PIXEL DUONG TINH THAT trong mask cua tap NGUON
(khong can label dich - chi can label nguon, von co san).

Ky vong: domain gap cang lon -> ty le du doan tren dich cang SUP xuong gan 0
so voi ty le "binh thuong" o nguon.
"""
import os
import yaml
from glob import glob

import torch
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import archs
from dataset import Dataset

PAIRS = [
    ('chase_unet', 'hrf'),
    ('chase_unet', 'rite'),
    ('hrf_unet', 'chase'),
    ('hrf_unet', 'rite'),
]

# ten thu muc dataset ung voi tung source model - de do ty le mask that o NGUON
SOURCE_DATASET = {
    'chase_unet': 'chase',
    'hrf_unet': 'hrf',
}


@torch.no_grad()
def predicted_positive_ratio(model, loader):
    model.eval()
    total_ratio, n = 0.0, 0
    for input, _, _ in loader:
        input = input.cuda()
        output = model(input)
        prob = torch.sigmoid(output)
        ratio = (prob > 0.5).float().mean().item()
        total_ratio += ratio * input.size(0)
        n += input.size(0)
    return total_ratio / n


def gt_positive_ratio(mask_dir):
    """Ty le pixel duong tinh THAT trong mask ground-truth (dung cho tap nguon,
    khong dung label dich - chi minh hoa 'ty le binh thuong' can co)."""
    import cv2
    import numpy as np
    files = glob(os.path.join(mask_dir, '*'))
    total, n = 0.0, 0
    for f in files:
        m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        total += (m > 127).mean()
        n += 1
    return total / n


def main():
    print(f"{'Source':<12} {'Target':<8} {'Ty le du doan (dich)':<22} "
          f"{'Ty le that (nguon)':<20} {'Ty le tuong doi':<15}")
    print("-" * 80)

    for source, target in PAIRS:
        cfg_path = f'models/{source}/config_{target}.yml'
        with open(cfg_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ])

        img_ids = glob(os.path.join('inputs', target, 'train', 'images', '*' + config['img_ext']))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        dataset = Dataset(
            img_ids=img_ids,
            img_dir=os.path.join('inputs', target, 'train', 'images'),
            mask_dir=os.path.join('inputs', target, 'train', 'masks'),
            img_ext=config['img_ext'], mask_ext=config['mask_ext'],
            num_classes=config['num_classes'], transform=val_transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

        model = archs.__dict__[config['arch']](config['num_classes'],
                                                config['input_channels'],
                                                config['deep_supervision'])
        model.load_state_dict(torch.load(f'models/{source}/model.pth'))
        model.cuda()

        pred_ratio = predicted_positive_ratio(model, loader)

        src_name = SOURCE_DATASET[source]
        gt_ratio = gt_positive_ratio(f'inputs/{src_name}/train/masks/0')

        relative = pred_ratio / (gt_ratio + 1e-8)
        print(f"{source:<12} {target:<8} {pred_ratio:<22.4f} {gt_ratio:<20.4f} {relative:<15.4f}")


if __name__ == '__main__':
    main()
