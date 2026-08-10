"""
unsupervised_selector_v5b_audit.py
=====================================
Doc target TEST masks - CHI dung o day, SAU KHI unsupervised_selector_v5b.py
da chay xong va decision JSON da khoa. Tinh Dice that cho MOI (clahe,
resolution) da test, roi doi chieu voi lua chon cua tung lambda.

Cach dung:
    python run.py unsupervised_selector_v5b_audit.py --source chase_unet --target hrf \\
        --decision_json selector_v5b_outputs/chase_to_hrf_decision.json \\
        --resolutions 384 512 768 1024 \\
        --out_csv selector_v5b_audit/chase_to_hrf_audit.csv
"""
import os
import json
import argparse
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch

from unsupervised_selector_v5b import load_model, apply_clahe_lab, predict_with_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--decision_json', required=True)
    p.add_argument('--resolutions', type=int, nargs='+', default=[384, 512, 768, 1024])
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--out_csv', required=True)
    return p.parse_args()


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    den = pred.sum() + gt.sum()
    return 1.0 if den == 0 else float(2.0 * np.logical_and(pred, gt).sum() / den)


def main():
    args = parse_args()
    with open(args.decision_json) as f:
        d = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = load_model(args.source, args.target, device)

    img_dir = os.path.join('inputs', args.target, 'test', 'images')
    mask_dir = os.path.join('inputs', args.target, 'test', 'masks', '0')
    paths = sorted(glob(os.path.join(img_dir, '*' + cfg['img_ext'])))

    rows = []
    for clahe in [False, True]:
        for res in args.resolutions:
            vals = []
            for pth in paths:
                iid = os.path.splitext(os.path.basename(pth))[0]
                mpath = os.path.join(mask_dir, iid + cfg['mask_ext'])
                img = cv2.imread(pth, cv2.IMREAD_COLOR)
                gt = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if img is None or gt is None:
                    continue
                proc = apply_clahe_lab(img) if clahe else img
                prob, _ = predict_with_features(model, proc, res, device)
                h, w = img.shape[:2]
                prob_native = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
                gt_native = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
                vals.append(dice(prob_native >= args.threshold, gt_native > 127))
            rows.append({'resolution': res, 'clahe': clahe,
                         'dice_mean': float(np.mean(vals)), 'dice_std': float(np.std(vals)),
                         'n': len(vals)})

    df = pd.DataFrame(rows).sort_values('dice_mean', ascending=False).reset_index(drop=True)
    oracle = df.iloc[0]

    print(df.to_string(index=False))
    print(f"\nZero-shot oracle: {int(oracle['resolution'])}px, clahe={bool(oracle['clahe'])}, "
          f"Dice={oracle['dice_mean']:.4f}")

    print(f"\nCLAHE V5b chon: {d['clahe']}")
    print("\nRegret theo tung lambda:")
    regret_rows = []
    for lam, res in d['resolution_by_lambda'].items():
        sel = df[(df['resolution'] == int(res)) & (df['clahe'] == bool(d['clahe']))]
        if len(sel) == 0:
            print(f"  lambda={lam}: resolution={res} - KHONG TIM THAY trong audit (kiem tra --resolutions)")
            continue
        sel_dice = float(sel.iloc[0]['dice_mean'])
        regret = float(oracle['dice_mean'] - sel_dice)
        print(f"  lambda={lam}: resolution={res}, Dice={sel_dice:.4f}, regret={regret:.4f}")
        regret_rows.append({'lambda': lam, 'resolution': res, 'dice': sel_dice, 'regret': regret})

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    pd.DataFrame(regret_rows).to_csv(args.out_csv.replace('.csv', '_regret_by_lambda.csv'), index=False)


if __name__ == '__main__':
    main()
