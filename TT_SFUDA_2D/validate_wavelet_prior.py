"""
validate_wavelet_prior.py
============================
Validate wavelet_vesselness() DOC LAP tren 3 dataset (CHASE/HRF/RITE),
DUNG Y HET quy trinh da lam voi Frangi truoc day: so Dice TRUC TIEP voi
ground-truth, threshold quet qua nhieu gia tri (KHONG dung 1 nguong co
dinh doan mo - day la diem sua so voi Frangi, noi threshold 0.005 co dinh
tren toan dataset la nguyen nhan gay -0.83 diem khi ep cung sau nay).

Doc anh + CLAHE GIONG HET frangi_prior.compute_frangi_map (kenh green,
CLAHE clip=2.0, tile=(8,8)) de so sanh cong bang - khac bien Dice (neu co)
phan anh dung KHAC BIET giua 2 phuong phap, khong phai do tien xu ly khac nhau.

KET QUA THAM CHIEU (Frangi, DA CO SAN, threshold co dinh 0.005):
  RITE (n=5):  Dice = 0.5352 +/- 0.052
  CHASE (n=5): Dice = 0.4423 +/- 0.047
  HRF (n=5):   Dice = 0.4258 +/- 0.059
  Gop ca 3 (n=15): Dice = 0.4678 +/- 0.072

Cach dung:
    python run.py validate_wavelet_prior.py --n_images 5
    python run.py validate_wavelet_prior.py --n_images 10 --dataset rite
"""
import os
import argparse
from glob import glob

import cv2
import numpy as np
import yaml

from wavelet_prior import wavelet_vesselness, evaluate_standalone_dice

# Config bat ky co target = dataset can validate, chi de lay img_ext/mask_ext
# (giong nhau giua cac source cho cung 1 target dataset).
DATASET_CONFIG_SOURCE = {
    'chase': ('hrf_unet', 'config_chase_dualema'),
    'hrf': ('chase_unet', 'config_hrf_dualema'),
    'rite': ('chase_unet', 'config_rite_dualema'),
}

FRANGI_REFERENCE = {
    'rite': 0.5352, 'chase': 0.4423, 'hrf': 0.4258, 'combined': 0.4678,
}

THRESHOLDS_TO_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_images', type=int, default=5,
                         help='So anh moi dataset, mac dinh 5 (khop so luong da dung voi Frangi).')
    parser.add_argument('--dataset', default=None,
                         help='Chi validate 1 dataset (chase/hrf/rite). Mac dinh: ca 3.')
    parser.add_argument('--wavelet_type', default='db4')
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--percentile_threshold', type=float, default=92.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_config_for_dataset(dataset_name):
    source, config_file = DATASET_CONFIG_SOURCE[dataset_name]
    with open(f'models/{source}/{config_file}.yml', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def read_green_clahe(img_path, clahe_clip=2.0, clahe_tile=(8, 8)):
    """Doc anh + lay kenh green + CLAHE - GIONG HET frangi_prior.compute_frangi_map
    de dam bao so sanh cong bang (khac Dice phan anh dung khac biet phuong phap)."""
    img = cv2.imread(img_path)
    green = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    green_clahe = clahe.apply(green)
    return green_clahe.astype(np.float64) / 255.0


def validate_one_dataset(dataset_name, n_images, wavelet_kwargs, seed):
    config = load_config_for_dataset(dataset_name)
    img_dir = os.path.join('inputs', dataset_name, 'train', 'images')
    mask_dir = os.path.join('inputs', dataset_name, 'train', 'masks', '0')

    img_paths = sorted(glob(os.path.join(img_dir, '*' + config['img_ext'])))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(img_paths), size=min(n_images, len(img_paths)), replace=False)
    img_paths = [img_paths[i] for i in chosen]

    # dice_per_threshold[t] = list Dice tren tung anh, de tim threshold TOT
    # NHAT sau khi xem het du lieu (khac Frangi - noi 0.005 duoc CHON TRUOC
    # tu 1 lan thu, khong sweep). O day sweep RO RANG, minh bach.
    dice_per_threshold = {t: [] for t in THRESHOLDS_TO_SWEEP}
    per_image_best = []

    for img_path in img_paths:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, img_id + config['mask_ext'])
        if not os.path.exists(mask_path):
            print(f"  [BO QUA] khong tim thay mask cho {img_id}")
            continue

        image_gray = read_green_clahe(img_path)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt, (image_gray.shape[1], image_gray.shape[0]))
        gt_binary = (gt_resized > 127).astype(np.uint8)

        vmap = wavelet_vesselness(image_gray, **wavelet_kwargs)

        best_dice_this_img, best_t = -1.0, None
        for t in THRESHOLDS_TO_SWEEP:
            d = evaluate_standalone_dice(vmap, gt_binary, threshold=t)
            dice_per_threshold[t].append(d)
            if d > best_dice_this_img:
                best_dice_this_img, best_t = d, t
        per_image_best.append((img_id, best_dice_this_img, best_t))
        print(f"    {img_id}: best Dice = {best_dice_this_img:.4f} (threshold={best_t})")

    mean_per_threshold = {t: (np.mean(v) if v else float('nan')) for t, v in dice_per_threshold.items()}
    best_global_t = max(mean_per_threshold, key=mean_per_threshold.get)
    best_global_mean = mean_per_threshold[best_global_t]
    best_global_std = np.std(dice_per_threshold[best_global_t]) if dice_per_threshold[best_global_t] else float('nan')

    return {
        'dataset': dataset_name,
        'n_images': len(per_image_best),
        'mean_per_threshold': mean_per_threshold,
        'best_threshold': best_global_t,
        'best_mean_dice': best_global_mean,
        'best_std_dice': best_global_std,
    }


def main():
    args = parse_args()
    wavelet_kwargs = dict(wavelet=args.wavelet_type, levels=args.levels,
                           percentile_threshold=args.percentile_threshold)

    datasets = [args.dataset] if args.dataset else ['chase', 'hrf', 'rite']

    print(f"{'Dataset':<10} {'N':<4} {'Best threshold':<16} {'Wavelet Dice':<18} {'Frangi (tham chieu)':<20}")
    print("-" * 75)

    all_results = []
    for ds in datasets:
        print(f"\n[{ds.upper()}] Dang tinh wavelet_vesselness cho {args.n_images} anh...")
        result = validate_one_dataset(ds, args.n_images, wavelet_kwargs, args.seed)
        all_results.append(result)
        print(f"{ds:<10} {result['n_images']:<4} {result['best_threshold']:<16} "
              f"{result['best_mean_dice']:.4f} +/- {result['best_std_dice']:.4f}   "
              f"{FRANGI_REFERENCE.get(ds, float('nan')):<20.4f}")

    if len(all_results) > 1:
        combined_mean = np.mean([r['best_mean_dice'] for r in all_results])
        print("\n" + "=" * 75)
        print(f"TRUNG BINH GOP CA {len(all_results)} DATASET: wavelet = {combined_mean:.4f}  "
              f"| Frangi (tham chieu) = {FRANGI_REFERENCE['combined']:.4f}")
        if combined_mean > FRANGI_REFERENCE['combined']:
            print(f"  -> Wavelet VUOT Frangi doc lap (+{combined_mean - FRANGI_REFERENCE['combined']:.4f} diem). "
                  f"Dang thu tiep buoc tich hop mem qua wavelet_vote_weight() voi alpha nho (0.2-0.3).")
        else:
            print(f"  -> Wavelet KHONG vuot Frangi doc lap ({combined_mean - FRANGI_REFERENCE['combined']:+.4f} diem). "
                  f"Can can nhac: (a) dieu chinh tham so wavelet (levels, percentile_threshold) roi do lai, "
                  f"hoac (b) ket luan huong nay khong tot hon Frangi, dung lai o day, KHONG tich hop vao pipeline chinh.")
        print("=" * 75)


if __name__ == '__main__':
    main()
