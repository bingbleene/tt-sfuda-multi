"""
diagnose_gap.py
================
Do entropy trung binh cua Theta_s (source model, TRUOC khi adapt) tren anh
DICH CHUA GAN NHAN, cho ca 4 domain shift. Day la tin hieu duy nhat dung
duoc trong thuc te SFUDA (khong co label dich) de uoc luong domain gap.

Muc dich: xem entropy nay co TUONG QUAN voi "Direct Testing dice" (biet duoc
tu paper/da do o cac lan chay truoc) hay khong - neu co, dung lam co che
TU DONG chon topology (khong can nhin label).

Chay: dat trong TT_SFUDA_2D/, sau khi da co inputs/ va models/ (Cell 2,3):
    python run.py diagnose_gap.py
"""
import os
import yaml
from glob import glob

import torch
import torch.nn.functional as F
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


@torch.no_grad()
def average_entropy(model, loader):
    model.eval()
    total_ent, n = 0.0, 0
    for input, _, _ in loader:
        input = input.cuda()
        output = model(input)
        prob = torch.sigmoid(output)
        eps = 1e-8
        ent = -(prob * torch.log(prob + eps) + (1 - prob) * torch.log(1 - prob + eps))
        total_ent += ent.mean().item() * input.size(0)
        n += input.size(0)
    return total_ent / n


def main():
    print(f"{'Source':<12} {'Target':<8} {'Entropy TB (Theta_s tren dich)':<30}")
    print("-" * 55)

    for source, target in PAIRS:
        # dung config baseline (khong phai dualema) vi chi can img_ext/input_h/w/arch
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

        ent = average_entropy(model, loader)
        print(f"{source:<12} {target:<8} {ent:<30.4f}")


if __name__ == '__main__':
    main()
