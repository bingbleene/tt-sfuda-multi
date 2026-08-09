"""
validate_sam_prior.py
========================
Validate SAM vanilla (khong fine-tune) DOC LAP tren CHASE/HRF/RITE, dung
DUNG quy trinh da lam voi Frangi va wavelet: so Dice TRUC TIEP voi GT,
quet threshold, so sanh truc tiep voi 2 con so da co.

KY VONG THAP la hop ly (xem canh bao trong sam_prior.py) - day la buoc
kiem tra RE truoc khi quyet dinh co dang fine-tune SAM hay khong (fine-tune
ton nhieu thoi gian hon nhieu, ngoai pham vi 1 pilot).

Cach dung:
    pip install git+https://github.com/facebookresearch/segment-anything.git --break-system-packages
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
    python run.py validate_sam_prior.py --sam_checkpoint checkpoints/sam_vit_b_01ec64.pth --n_images 5
"""
import os
import argparse
from glob import glob

import cv2
import numpy as np
import yaml

from sam_prior import build_sam_predictor, compute_sam_vesselness_map
from wavelet_prior import evaluate_standalone_dice  # tai dung, khong viet lai ham Dice

DATASET_CONFIG_SOURCE = {
    'chase': ('hrf_unet', 'config_chase_dualema'),
    'hrf': ('chase_unet', 'config_hrf_dualema'),
    'rite': ('chase_unet', 'config_rite_dualema'),
}

REFERENCE_DICE = {
    'wavelet': {'rite': 0.3400, 'chase': 0.2716, 'hrf': 0.1975, 'combined': 0.2697},
    'frangi': {'rite': 0.5352, 'chase': 0.4423, 'hrf': 0.4258, 'combined': 0.4678},
}

THRESHOLDS_TO_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_checkpoint', required=True)
    parser.add_argument('--sam_model_type', default='vit_b')
    parser.add_argument('--n_images', type=int, default=5)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--points_per_side', type=int, default=16,
                         help='16 -> 256 diem prompt/anh. Tang len se cham hon nhieu '
                              '(vd 32 -> 1024 diem) - bat dau nho de test nhanh truoc.')
    parser.add_argument('--min_score', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def load_config_for_dataset(dataset_name):
    source, config_file = DATASET_CONFIG_SOURCE[dataset_name]
    with open(f'models/{source}/{config_file}.yml', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def validate_one_dataset(dataset_name, predictor, n_images, points_per_side, min_score, seed):
    config = load_config_for_dataset(dataset_name)
    img_dir = os.path.join('inputs', dataset_name, 'train', 'images')
    mask_dir = os.path.join('inputs', dataset_name, 'train', 'masks', '0')

    img_paths = sorted(glob(os.path.join(img_dir, '*' + config['img_ext'])))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(img_paths), size=min(n_images, len(img_paths)), replace=False)
    img_paths = [img_paths[i] for i in chosen]

    dice_per_threshold = {t: [] for t in THRESHOLDS_TO_SWEEP}

    for img_path in img_paths:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, img_id + config['mask_ext'])
        if not os.path.exists(mask_path):
            print(f"  [BO QUA] khong tim thay mask cho {img_id}")
            continue

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt, (img_rgb.shape[1], img_rgb.shape[0]))
        gt_binary = (gt_resized > 127).astype(np.uint8)

        vmap = compute_sam_vesselness_map(predictor, img_rgb,
                                           points_per_side=points_per_side,
                                           min_score=min_score)

        best_dice_this_img, best_t = -1.0, None
        for t in THRESHOLDS_TO_SWEEP:
            d = evaluate_standalone_dice(vmap, gt_binary, threshold=t)
            dice_per_threshold[t].append(d)
            if d > best_dice_this_img:
                best_dice_this_img, best_t = d, t
        print(f"    {img_id}: best Dice = {best_dice_this_img:.4f} (threshold={best_t}), "
              f"ty le pixel SAM tin cay (>0.5) = {(vmap > 0.5).mean():.4f}")

    mean_per_threshold = {t: (np.mean(v) if v else float('nan')) for t, v in dice_per_threshold.items()}
    best_global_t = max(mean_per_threshold, key=mean_per_threshold.get)
    best_global_mean = mean_per_threshold[best_global_t]
    best_global_std = np.std(dice_per_threshold[best_global_t]) if dice_per_threshold[best_global_t] else float('nan')

    return {
        'dataset': dataset_name,
        'best_threshold': best_global_t,
        'best_mean_dice': best_global_mean,
        'best_std_dice': best_global_std,
    }


def main():
    args = parse_args()
    print(f"[SETUP] Nap SAM ({args.sam_model_type}) tu {args.sam_checkpoint}...")
    predictor = build_sam_predictor(args.sam_checkpoint, args.sam_model_type, args.device)

    datasets = [args.dataset] if args.dataset else ['chase', 'hrf', 'rite']

    print(f"\n{'Dataset':<10} {'SAM Dice':<20} {'Frangi (tham chieu)':<20} {'Wavelet (tham chieu)':<20}")
    print("-" * 80)

    all_results = []
    for ds in datasets:
        print(f"\n[{ds.upper()}] Dang chay SAM ({args.points_per_side**2} diem prompt/anh, "
              f"{args.n_images} anh)...")
        result = validate_one_dataset(ds, predictor, args.n_images, args.points_per_side,
                                       args.min_score, args.seed)
        all_results.append(result)
        print(f"{ds:<10} {result['best_mean_dice']:.4f} +/- {result['best_std_dice']:.4f}      "
              f"{REFERENCE_DICE['frangi'].get(ds, float('nan')):<20.4f} "
              f"{REFERENCE_DICE['wavelet'].get(ds, float('nan')):<20.4f}")

    if len(all_results) > 1:
        combined_mean = np.mean([r['best_mean_dice'] for r in all_results])
        print("\n" + "=" * 80)
        print(f"TRUNG BINH GOP: SAM = {combined_mean:.4f}  |  Frangi = {REFERENCE_DICE['frangi']['combined']:.4f}"
              f"  |  Wavelet = {REFERENCE_DICE['wavelet']['combined']:.4f}")
        if combined_mean > REFERENCE_DICE['frangi']['combined']:
            print("  -> SAM VUOT Frangi ngay ca zero-shot - dang gia. Can nhac fine-tune tiep.")
        elif combined_mean > REFERENCE_DICE['wavelet']['combined']:
            print("  -> SAM hon wavelet nhung KHONG hon Frangi - dung o day, KHONG dang dau tu "
                  "fine-tune (Frangi da la lua chon tot hon, mien phi, khong can GPU lon).")
        else:
            print("  -> SAM THUA CA Frangi LAN wavelet - dong huong nay hoan toan, dung ke ca "
                  "y dinh fine-tune. Dung lai o day.")
        print("=" * 80)


if __name__ == '__main__':
    main()
