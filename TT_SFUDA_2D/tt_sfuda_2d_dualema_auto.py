"""
tt_sfuda_2d_dualema_auto.py
=============================
Tu dong chon topology + slow_keep_rate DUA TREN TIN HIEU KHONG CAN LABEL
DICH: ty le pixel du doan duong tinh cua Theta_s tren anh dich, so voi ty
le pixel duong tinh that trong mask NGUON (nguon co san label, khong vi
pham nguyen tac source-free vi khong dung label NGUON trong luc adapt,
chi dung TRUOC de uoc luong gap).

Nguong = 0.5 (hieu chinh tu diagnose_gap_v2.py tren 4 domain shift):
    ty_le_tuong_doi >= 0.5  -> gap NHO  -> cascaded, slow_keep_rate=0.999
    ty_le_tuong_doi <  0.5  -> gap LON  -> parallel, slow_keep_rate=0.995

Cach chay (thay the hoan toan cho viec tu chon --topology/--slow_keep_rate
bang tay):
    python run.py tt_sfuda_2d_dualema_auto.py --source chase_unet --target rite
"""
import os
import sys
import yaml
import argparse
import subprocess
from glob import glob

import cv2
import numpy as np
import torch
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import archs
from dataset import Dataset

GAP_THRESHOLD = 0.5
SOURCE_DATASET = {'chase_unet': 'chase', 'hrf_unet': 'hrf', 'rite_unet': 'rite'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--results_csv', default='results_dualema.csv')
    return p.parse_args()


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
    files = glob(os.path.join(mask_dir, '*'))
    total = 0.0
    for f in files:
        m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        total += (m > 127).mean()
    return total / len(files)


def estimate_gap_and_decide(source, target):
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
    gt_ratio = gt_positive_ratio(f"inputs/{SOURCE_DATASET[source]}/train/masks/0")
    relative = pred_ratio / (gt_ratio + 1e-8)

    if relative >= GAP_THRESHOLD:
        decision = dict(topology='cascaded', slow_keep_rate=0.999, ensemble_mode='mean')
        gap_label = 'NHO'
    else:
        decision = dict(topology='parallel', slow_keep_rate=0.995, ensemble_mode='mean')
        gap_label = 'LON'

    print(f"[AUTO] {source} -> {target}: ty le tuong doi = {relative:.4f} "
          f"(nguong={GAP_THRESHOLD}) => domain gap UOC LUONG = {gap_label}")
    print(f"[AUTO] Quyet dinh: topology={decision['topology']}, "
          f"slow_keep_rate={decision['slow_keep_rate']}, ensemble_mode={decision['ensemble_mode']}")
    return decision


def main():
    args = parse_args()
    decision = estimate_gap_and_decide(args.source, args.target)

    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', args.source, '--target', args.target,
           '--topology', decision['topology'],
           '--ensemble_mode', decision['ensemble_mode'],
           '--slow_keep_rate', str(decision['slow_keep_rate']),
           '--results_csv', args.results_csv]
    print(f"[AUTO] Chay: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
