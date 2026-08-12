"""
check_native_contrast.py
===========================
Kiem tra gia thuyet RE: "RITE can CLAHE bat ke nguon nao, CHASE/HRF khong
can" co the giai thich DON GIAN boi do tuong phan GOC cua chinh target,
khong lien quan domain gap nguon-dich.

Neu dung, day la tieu chi CLAHE RE NHAT co the co: khong can source model,
khong can inference, khong can adapt - chi do 1 vai thong ke co ban tren
anh RAW cua target (RMS contrast, Michelson contrast, do lech chuan kenh
xanh - kenh thuong dung nhat cho fundus/vessel).

Cach dung (chay trong TT_SFUDA_2D/, khong can GPU):
    python run.py check_native_contrast.py --n_images 15
"""
import os
import argparse
from glob import glob

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_images', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


DATASETS = ['chase', 'hrf', 'rite']


def compute_contrast_stats(img_bgr):
    """Vai chi so tuong phan co ban, khong can model/nhan."""
    green = img_bgr[:, :, 1].astype(np.float64)  # kenh xanh - chuan cho fundus

    rms_contrast = float(green.std())

    p1, p99 = np.percentile(green, [1, 99])
    michelson = float((p99 - p1) / (p99 + p1 + 1e-8))

    # Tuong phan cuc bo (trung binh do lech chuan tren cua so 16x16)
    h, w = green.shape
    local_stds = []
    step = 16
    for y in range(0, h - step, step):
        for x in range(0, w - step, step):
            patch = green[y:y + step, x:x + step]
            local_stds.append(patch.std())
    local_contrast = float(np.mean(local_stds)) if local_stds else 0.0

    return {'rms_contrast': rms_contrast, 'michelson_contrast': michelson,
            'local_contrast': local_contrast}


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    results = {}
    for name in DATASETS:
        img_dir = os.path.join('inputs', name, 'train', 'images')
        paths = sorted(glob(os.path.join(img_dir, '*')))
        paths = [p for p in paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        if not paths:
            print(f"[THIEU] khong tim thay anh trong {img_dir}")
            continue

        n = min(args.n_images, len(paths))
        idx = rng.choice(len(paths), size=n, replace=False)
        chosen = [paths[i] for i in idx]

        stats_list = []
        for p in chosen:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            stats_list.append(compute_contrast_stats(img))

        agg = {}
        for key in ['rms_contrast', 'michelson_contrast', 'local_contrast']:
            vals = [s[key] for s in stats_list]
            agg[key] = (float(np.mean(vals)), float(np.std(vals)))
        results[name] = agg

        print(f"\n{name} (n={len(stats_list)} anh):")
        for key, (m, s) in agg.items():
            print(f"  {key}: {m:.3f} +/- {s:.3f}")

    print(f"\n{'='*60}\nSO SANH TRUC TIEP")
    print(f"{'':<12}", end='')
    for name in results:
        print(f"{name:<20}", end='')
    print()
    for key in ['rms_contrast', 'michelson_contrast', 'local_contrast']:
        print(f"{key:<12}", end='')
        for name in results:
            m, s = results[name][key]
            print(f"{m:<20.3f}", end='')
        print()

    print(f"\n{'='*60}")
    if 'rite' in results and 'chase' in results and 'hrf' in results:
        rite_rms = results['rite']['rms_contrast'][0]
        chase_rms = results['chase']['rms_contrast'][0]
        hrf_rms = results['hrf']['rms_contrast'][0]
        if rite_rms < chase_rms and rite_rms < hrf_rms:
            print(f"=> XAC NHAN: RITE co RMS contrast THAP NHAT ({rite_rms:.3f} so voi "
                  f"CHASE={chase_rms:.3f}, HRF={hrf_rms:.3f}) - UNG HO gia thuyet "
                  f"'do tuong phan goc thap la ly do RITE can CLAHE'.")
        else:
            print(f"=> KHONG xac nhan: RITE khong phai dataset co contrast thap nhat "
                  f"(RITE={rite_rms:.3f}, CHASE={chase_rms:.3f}, HRF={hrf_rms:.3f}) - "
                  f"gia thuyet do tuong phan don gian CHUA giai thich duoc, can huong khac.")


if __name__ == '__main__':
    main()
