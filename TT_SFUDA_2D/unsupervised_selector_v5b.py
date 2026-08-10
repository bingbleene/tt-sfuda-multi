"""
unsupervised_selector_v5b.py
==============================
V5b - thay V5 (da bi loai vi hook BN chet). Giu nguyen 2 phan da kiem
chung/thong nhat:

  A. CLAHE quyet dinh qua Response Recovery cua V3 (TTA-flip stability +
     Frangi lam tin hieu doc lap + background-explosion penalty) - phan
     nay DA hoat dong tot (regret=0 cho ca 2 shift CHASE-nguon), giu nguyen.

  B. Resolution quyet dinh qua FEATURE-CONSISTENCY THAT (khac han V5-BN):
     dung model(x, mode='const') - INTERFACE DA CO SAN, DA DUNG NHIEU LAN
     TRONG DU AN (consistency_loss, cross_arch_bottleneck...), tra ve dung
     4 tang feature [x1_0,x2_0,x3_0,x4_0] THAT SU CHAY QUA FORWARD (khac
     BN o V5 - khong con nguy co "hook khong bao gio duoc goi").

     Ket hop them geometry_prior.py (S = S_feature + lambda*S_geometry) -
     lambda quet qua nhieu gia tri, KHONG co dinh 1 gia tri de bao ve truoc
     hoi dong (xem ky hon o audit/ablation rieng).

KHONG doc target masks/source images o buoc quyet dinh. Target test masks
CHI duoc doc o script audit rieng (unsupervised_selector_v5b_audit.py),
SAU KHI decision JSON da khoa.

Cach dung:
    python run.py unsupervised_selector_v5b.py --source chase_unet --target hrf \\
        --resolutions 384 512 768 1024 --n_images 10 --bootstrap 100 \\
        --lambdas 0.0 0.3 0.5 1.0 --out_prefix selector_v5b_outputs/chase_to_hrf
"""
import os
import json
import argparse
from glob import glob

import cv2
import yaml
import numpy as np
import pandas as pd
import torch

import archs
from patch_inference import normalize_like_dataset
from geometry_prior import combined_marginal_utility

try:
    from skimage.filters import frangi
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--resolutions', type=int, nargs='+', default=[384, 512, 768, 1024])
    p.add_argument('--n_images', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--bootstrap', type=int, default=100)
    p.add_argument('--diag_size', type=int, default=512)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--recovery_delta', type=float, default=0.15)
    p.add_argument('--frangi_threshold', type=float, default=0.01)
    p.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.3, 0.5, 1.0],
                    help='Quet nhieu lambda cho geometry prior - KHONG chon 1 gia tri '
                         'co dinh truoc, de audit quyet dinh lambda nao dang tin.')
    p.add_argument('--stop_drop_ratio', type=float, default=0.85,
                    help='Dung tang resolution khi combined utility cua transition hien '
                         'tai < ty le nay nhan max utility da thay - nguong tam thoi, '
                         'CAN dieu chinh sau khi xem ket qua thuc te, khong phai chan ly.')
    p.add_argument('--out_prefix', required=True)
    return p.parse_args()


def apply_clahe_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = c.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def load_model(source, target, device):
    cfg_path = f'models/{source}/config_{target}_dualema.yml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = archs.__dict__[cfg['arch']](cfg['num_classes'], cfg['input_channels'],
                                         cfg['deep_supervision']).to(device)
    state = torch.load(f'models/{source}/model.pth', map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_with_features(model, img_bgr, res, device):
    """Tra ve (prob_map, [feat1,feat2,feat3,feat4]) - dung DUNG interface
    mode='const' da co san trong archs.UNet, KHONG tu hook gi ca."""
    ximg = cv2.resize(img_bgr, (res, res), interpolation=cv2.INTER_LINEAR)
    ximg = normalize_like_dataset(ximg)
    x = torch.from_numpy(ximg.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    logits, feats = model(x, mode='const')
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
    # Global-average-pool moi tang feature ve 1 vector channel - so sanh duoc
    # giua cac resolution khac nhau du spatial size khac nhau.
    pooled = [f.mean(dim=[2, 3])[0].cpu().numpy() for f in feats]
    return prob, pooled


def cosine_sim(a, b, eps=1e-8):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def resize01(x, size):
    return cv2.resize(x.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def frangi_map(img_bgr, diag_size):
    green = img_bgr[:, :, 1]
    green = cv2.resize(green, (diag_size, diag_size), interpolation=cv2.INTER_AREA)
    inv = 1.0 - green.astype(np.float32) / 255.0
    if HAS_SKIMAGE:
        v = frangi(inv, sigmas=range(1, 5), black_ridges=False)
        v = np.nan_to_num(v).astype(np.float32)
    else:
        b1 = cv2.GaussianBlur(inv, (0, 0), 1.0)
        b3 = cv2.GaussianBlur(inv, (0, 0), 3.0)
        v = np.maximum(b1 - b3, 0.0)
    vmax = float(v.max())
    return v / vmax if vmax > 0 else v


def recovery_metrics(raw_p, clahe_p, raw_pf, clahe_pf, frangi_v,
                      threshold, recovery_delta, frangi_threshold):
    recovered = (raw_p < threshold) & ((clahe_p - raw_p) >= recovery_delta)
    recovered_flip = (raw_pf < threshold) & ((clahe_pf - raw_pf) >= recovery_delta)
    stable_recovery = recovered & recovered_flip
    vessel_like = frangi_v >= frangi_threshold

    recovered_ratio = float(recovered.mean())
    stable_ratio = float(stable_recovery.mean())
    if recovered.sum() > 0:
        recovered_vessel_like = float(vessel_like[recovered].mean())
        recovered_stable_fraction = float(stable_recovery[recovered].mean())
    else:
        recovered_vessel_like = 0.0
        recovered_stable_fraction = 0.0

    new_positive = (raw_p < threshold) & (clahe_p >= threshold)
    background_explosion = float((~vessel_like[new_positive]).mean()) if new_positive.sum() > 0 else 0.0

    fg_raw = float((raw_p >= threshold).mean())
    fg_clahe = float((clahe_p >= threshold).mean())

    return {
        'recovered_ratio': recovered_ratio, 'stable_recovery_ratio': stable_ratio,
        'recovered_vessel_like': recovered_vessel_like,
        'recovered_stable_fraction': recovered_stable_fraction,
        'background_explosion': background_explosion,
        'fg_gain': fg_clahe - fg_raw,
    }


def summarize(records):
    out = {}
    for k in records[0].keys():
        vals = np.asarray([r[k] for r in records], dtype=np.float64)
        out[k + '_mean'] = float(vals.mean())
        out[k + '_std'] = float(vals.std())
    return out


def appearance_score(row):
    useful_recovery = (row['recovered_ratio_mean'] * row['recovered_vessel_like_mean']
                        * row['recovered_stable_fraction_mean'])
    fg_gain = row['fg_gain_mean']
    fg_term = min(fg_gain / 0.05, 1.0) if fg_gain > 0 else 0.0
    penalty = row['background_explosion_mean']
    return 0.55 * useful_recovery * 20.0 + 0.20 * row['stable_recovery_ratio_mean'] * 20.0 + 0.15 * fg_term - 0.25 * penalty


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[DEVICE]', device)

    model, cfg = load_model(args.source, args.target, device)
    target_dataset = args.target  # ten dataset target ('chase'/'hrf'/'rite') - trung voi args.target

    img_dir = os.path.join('inputs', args.target, 'train', 'images')
    paths = sorted(glob(os.path.join(img_dir, '*' + cfg['img_ext'])))
    rng = np.random.RandomState(args.seed)
    if 0 < args.n_images < len(paths):
        idx = rng.choice(len(paths), size=args.n_images, replace=False)
        paths = [paths[i] for i in sorted(idx)]

    resolutions = sorted(args.resolutions)
    canonical = 512 if 512 in resolutions else resolutions[len(resolutions) // 2]
    print(f'[DATA] {args.source}->{args.target}, n={len(paths)}')
    print('[RESOLUTIONS]', resolutions)

    # ---- Thu thap prob + feature cho moi (clahe, resolution) ----
    probs, probs_flip, feats_by_key, frangis = {}, {}, {}, []
    for i, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(path)
        frangis.append(frangi_map(img, args.diag_size))
        print(f'[{i+1}/{len(paths)}] {os.path.basename(path)}')

        for clahe in [False, True]:
            proc = apply_clahe_lab(img) if clahe else img
            proc_f = cv2.flip(proc, 1)
            for res in resolutions:
                key = (clahe, res)
                probs.setdefault(key, [])
                probs_flip.setdefault(key, [])
                feats_by_key.setdefault(key, [])

                p, feat = predict_with_features(model, proc, res, device)
                pf, _ = predict_with_features(model, proc_f, res, device)
                pf = cv2.flip(pf, 1)

                probs[key].append(resize01(p, args.diag_size))
                probs_flip[key].append(resize01(pf, args.diag_size))
                feats_by_key[key].append(feat)

    # ---- A. CLAHE qua Response Recovery (V3, khong doi) ----
    recovery_records = []
    for i in range(len(paths)):
        recovery_records.append(recovery_metrics(
            probs[(False, canonical)][i], probs[(True, canonical)][i],
            probs_flip[(False, canonical)][i], probs_flip[(True, canonical)][i],
            frangis[i], args.threshold, args.recovery_delta, args.frangi_threshold))
    app = summarize(recovery_records)
    app['appearance_score'] = appearance_score(app)
    selected_clahe = bool(app['appearance_score'] > 0.0)

    # ---- B. Resolution qua feature-consistency + geometry (MOI) ----
    transition_rows = []
    for j in range(len(resolutions) - 1):
        r_small, r_large = resolutions[j], resolutions[j + 1]
        sims_per_image = []
        for i in range(len(paths)):
            feat_small = feats_by_key[(selected_clahe, r_small)][i]
            feat_large = feats_by_key[(selected_clahe, r_large)][i]
            level_sims = [cosine_sim(fs, fl) for fs, fl in zip(feat_small, feat_large)]
            sims_per_image.append(float(np.mean(level_sims)))
        feature_utility = float(np.mean(sims_per_image))

        row = {'res_small': r_small, 'res_large': r_large, 'feature_utility': feature_utility}
        for lam in args.lambdas:
            row[f'combined_lambda_{lam}'] = combined_marginal_utility(
                feature_utility, r_small, r_large, target_dataset, args.source, lam)
        transition_rows.append(row)

    trans_df = pd.DataFrame(transition_rows)

    # ---- Quyet dinh resolution cho MOI lambda (de so sanh) ----
    decisions_per_lambda = {}
    for lam in args.lambdas:
        col = f'combined_lambda_{lam}'
        max_seen = trans_df[col].iloc[0]
        selected_res = resolutions[0]
        for _, row in trans_df.iterrows():
            if row[col] < args.stop_drop_ratio * max_seen:
                selected_res = int(row['res_small'])
                break
            max_seen = max(max_seen, row[col])
            selected_res = int(row['res_large'])
        decisions_per_lambda[lam] = selected_res

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    trans_df.to_csv(args.out_prefix + '_transitions.csv', index=False)
    pd.DataFrame([app]).to_csv(args.out_prefix + '_appearance.csv', index=False)

    decision = {
        'source': args.source, 'target': args.target,
        'selector_version': 'V5b_feature_consistency_plus_geometry',
        'clahe': selected_clahe,
        'appearance_score': float(app['appearance_score']),
        'resolution_by_lambda': decisions_per_lambda,
        'lambdas_tested': list(args.lambdas),
        'uses_target_labels': False,
        'uses_source_images_or_masks': False,
        'stop_drop_ratio': args.stop_drop_ratio,
    }
    with open(args.out_prefix + '_decision.json', 'w') as f:
        json.dump(decision, f, indent=2)

    print('\n=== APPEARANCE (CLAHE) ===')
    print(f"appearance_score: {app['appearance_score']:.4f} -> clahe={selected_clahe}")

    print('\n=== FEATURE-CONSISTENCY + GEOMETRY TRANSITIONS ===')
    print(trans_df.to_string(index=False))

    print('\n=== RESOLUTION DA CHON THEO TUNG LAMBDA ===')
    for lam, res in decisions_per_lambda.items():
        print(f'  lambda={lam}: resolution={res}')

    print('\n=== V5b DECISION (JSON) ===')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
