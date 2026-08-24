
import os
import json
import argparse
from glob import glob

import cv2
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import archs
from patch_inference import normalize_like_dataset

try:
    from skimage.filters import frangi
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


ENCODER_BLOCKS = [
    'conv0_0',
    'conv1_0',
    'conv2_0',
    'conv3_0',
    'conv4_0',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--resolutions', type=int, nargs='+',
                   default=[384, 512, 768, 1024])
    p.add_argument('--n_images', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--bootstrap', type=int, default=300)
    p.add_argument('--diag_size', type=int, default=512)
    p.add_argument('--threshold', type=float, default=0.5)

    # V3 appearance branch
    p.add_argument('--recovery_delta', type=float, default=0.15)
    p.add_argument('--frangi_threshold', type=float, default=0.01)

    # V5b latent branch
    p.add_argument('--pool_size', type=int, default=16)
    p.add_argument('--min_feature_gain', type=float, default=0.0)
    p.add_argument('--overscale_drop', type=float, default=0.06)

    p.add_argument('--out_prefix', required=True)
    return p.parse_args()


def unwrap_output(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        for k in ['out', 'logits', 'seg', 'prediction']:
            if k in out:
                return out[k]
        return next(iter(out.values()))
    return out


def load_model(source, target, device):
    cfg_path = f'models/{source}/config_{target}_dualema.yml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = archs.__dict__[cfg['arch']](
        cfg['num_classes'],
        cfg['input_channels'],
        cfg['deep_supervision'],
    ).to(device)

    state = torch.load(
        f'models/{source}/model.pth',
        map_location=device
    )
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def apply_clahe_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    c = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    l2 = c.apply(l)
    return cv2.cvtColor(
        cv2.merge([l2, a, b]),
        cv2.COLOR_LAB2BGR
    )


def preprocess_tensor(img_bgr, res, device):
    ximg = cv2.resize(
        img_bgr,
        (res, res),
        interpolation=cv2.INTER_LINEAR
    )
    ximg = normalize_like_dataset(ximg)
    x = torch.from_numpy(
        ximg.transpose(2, 0, 1)
    ).float().unsqueeze(0).to(device)
    return x


@torch.no_grad()
def predict(model, img_bgr, res, device):
    x = preprocess_tensor(
        img_bgr,
        res,
        device
    )
    logits = unwrap_output(model(x))
    return torch.sigmoid(
        logits
    )[0, 0].cpu().numpy().astype(np.float32)


def resize01(x, size):
    return cv2.resize(
        x.astype(np.float32),
        (size, size),
        interpolation=cv2.INTER_LINEAR
    )


def frangi_map(img_bgr, diag_size):
    green = img_bgr[:, :, 1]
    green = cv2.resize(
        green,
        (diag_size, diag_size),
        interpolation=cv2.INTER_AREA
    )
    inv = 1.0 - green.astype(np.float32) / 255.0

    if HAS_SKIMAGE:
        v = frangi(
            inv,
            sigmas=range(1, 5),
            black_ridges=False
        )
        v = np.nan_to_num(v).astype(np.float32)
    else:
        b1 = cv2.GaussianBlur(inv, (0, 0), 1.0)
        b3 = cv2.GaussianBlur(inv, (0, 0), 3.0)
        v = np.maximum(b1 - b3, 0.0)

    vmax = float(v.max())
    if vmax > 0:
        v = v / vmax
    return v.astype(np.float32)


def recovery_metrics(raw_p, clahe_p, raw_pf, clahe_pf, frangi_v,
                     threshold, recovery_delta, frangi_threshold):
    recovered = (
        (raw_p < threshold) &
        ((clahe_p - raw_p) >= recovery_delta)
    )

    recovered_flip = (
        (raw_pf < threshold) &
        ((clahe_pf - raw_pf) >= recovery_delta)
    )

    stable_recovery = (
        recovered &
        recovered_flip
    )

    vessel_like = (
        frangi_v >= frangi_threshold
    )

    if recovered.sum() > 0:
        recovered_vessel_like = float(
            vessel_like[recovered].mean()
        )
        recovered_stable_fraction = float(
            stable_recovery[recovered].mean()
        )
    else:
        recovered_vessel_like = 0.0
        recovered_stable_fraction = 0.0

    new_positive = (
        (raw_p < threshold) &
        (clahe_p >= threshold)
    )

    if new_positive.sum() > 0:
        background_explosion = float(
            (~vessel_like[new_positive]).mean()
        )
    else:
        background_explosion = 0.0

    fg_raw = float(
        (raw_p >= threshold).mean()
    )
    fg_clahe = float(
        (clahe_p >= threshold).mean()
    )

    return {
        'recovered_ratio': float(
            recovered.mean()
        ),
        'stable_recovery_ratio': float(
            stable_recovery.mean()
        ),
        'recovered_vessel_like': recovered_vessel_like,
        'recovered_stable_fraction': recovered_stable_fraction,
        'background_explosion': background_explosion,
        'fg_raw': fg_raw,
        'fg_clahe': fg_clahe,
        'fg_gain': fg_clahe - fg_raw,
    }


def summarize(records):
    out = {}
    for k in records[0].keys():
        vals = np.asarray(
            [r[k] for r in records],
            dtype=np.float64
        )
        out[k + '_mean'] = float(vals.mean())
        out[k + '_std'] = float(vals.std())
    return out


def appearance_score(row):
    useful_recovery = (
        row['recovered_ratio_mean']
        * row['recovered_vessel_like_mean']
        * row['recovered_stable_fraction_mean']
    )

    fg_gain = row['fg_gain_mean']
    fg_term = (
        0.0
        if fg_gain <= 0
        else min(fg_gain / 0.05, 1.0)
    )

    return float(
        0.55 * useful_recovery * 20.0
        + 0.20 * row['stable_recovery_ratio_mean'] * 20.0
        + 0.15 * fg_term
        - 0.25 * row['background_explosion_mean']
    )


class FeatureCollector:
    def __init__(self, model, block_names):
        self.features = {}
        self.handles = []

        named = dict(
            model.named_modules()
        )

        missing = [
            x for x in block_names
            if x not in named
        ]
        if missing:
            raise RuntimeError(
                f'Missing forward blocks: {missing}'
            )

        for name in block_names:
            module = named[name]
            h = module.register_forward_hook(
                self._make_hook(name)
            )
            self.handles.append(h)

    def _make_hook(self, name):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.features[name] = out.detach()
        return hook

    def reset(self):
        self.features = {}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def collect_features(
    model,
    collector,
    img_bgr,
    res,
    device,
    pool_size
):
    collector.reset()

    x = preprocess_tensor(
        img_bgr,
        res,
        device
    )
    _ = model(x)

    out = {}

    for lname, feat in collector.features.items():
        if feat.ndim != 4:
            continue

        pooled = F.adaptive_avg_pool2d(
            feat,
            output_size=(pool_size, pool_size)
        )

        # Channel-wise normalization keeps scale comparison
        # from being dominated by raw amplitude.
        flat = pooled.flatten(2)
        ch_mean = flat.mean(dim=2, keepdim=True)
        ch_std = flat.std(
            dim=2,
            keepdim=True,
            unbiased=False
        ) + 1e-6
        norm = (
            (flat - ch_mean)
            / ch_std
        )

        # Representation summary
        vec = norm.flatten().cpu()

        energy = float(
            feat.pow(2).mean().sqrt().cpu()
        )

        positive_fraction = float(
            (feat > 0).float().mean().cpu()
        )

        # channel utilization:
        # fraction of channels with nontrivial mean absolute activation
        ch_mag = feat.abs().mean(
            dim=(0, 2, 3)
        )
        med = torch.median(ch_mag)
        utilization = float(
            (ch_mag > 0.1 * (med + 1e-8))
            .float().mean().cpu()
        )

        # effective rank on C x spatial pooled representation
        mat = norm[0]  # C x HW
        try:
            s = torch.linalg.svdvals(mat)
            p = s / (s.sum() + 1e-8)
            entropy = -torch.sum(
                p * torch.log(p + 1e-8)
            )
            eff_rank = float(
                torch.exp(entropy).cpu()
            )
        except Exception:
            eff_rank = float('nan')

        out[lname] = {
            'vec': vec,
            'energy': energy,
            'positive_fraction': positive_fraction,
            'utilization': utilization,
            'effective_rank': eff_rank,
        }

    return out


def cosine_from_vectors(a, b):
    a = a.float()
    b = b.float()

    if a.numel() != b.numel():
        raise RuntimeError(
            f'Feature vectors mismatch: {a.numel()} vs {b.numel()}'
        )

    return float(
        F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
            dim=1
        )[0].cpu()
    )


def transition_feature_metrics(
    f1,
    f2,
    layer_names
):
    per_layer = []

    for lname in layer_names:
        if (
            lname not in f1
            or lname not in f2
        ):
            continue

        a = f1[lname]
        b = f2[lname]

        cos = cosine_from_vectors(
            a['vec'],
            b['vec']
        )

        energy_change = (
            abs(
                b['energy']
                - a['energy']
            )
            / max(
                abs(a['energy']),
                1e-6
            )
        )

        util_change = abs(
            b['utilization']
            - a['utilization']
        )

        pos_change = abs(
            b['positive_fraction']
            - a['positive_fraction']
        )

        er1 = a['effective_rank']
        er2 = b['effective_rank']

        if (
            np.isfinite(er1)
            and np.isfinite(er2)
        ):
            rank_change = (
                abs(er2 - er1)
                / max(abs(er1), 1e-6)
            )
        else:
            rank_change = np.nan

        per_layer.append({
            'layer': lname,
            'cosine': cos,
            'energy_change': energy_change,
            'util_change': util_change,
            'positive_change': pos_change,
            'rank_change': rank_change,
        })

    if not per_layer:
        raise RuntimeError(
            'No valid latent feature pairs collected.'
        )

    df = pd.DataFrame(per_layer)

    return {
        'latent_cosine': float(
            df['cosine'].mean()
        ),
        'latent_cosine_min': float(
            df['cosine'].min()
        ),
        'energy_change': float(
            df['energy_change'].mean()
        ),
        'util_change': float(
            df['util_change'].mean()
        ),
        'positive_change': float(
            df['positive_change'].mean()
        ),
        'rank_change': float(
            np.nanmean(
                df['rank_change'].values
            )
        ),
    }, per_layer


def latent_marginal_score(row):
    """
    High is better:
    - high feature cosine consistency
    - low abrupt energy/utilization/positive/rank change

    This is not a Dice proxy; it is a latent stability score.
    """
    return float(
        0.50 * row['latent_cosine_mean']
        + 0.10 * row['latent_cosine_min_mean']
        + 0.10 * np.exp(-row['energy_change_mean'])
        + 0.10 * np.exp(-5.0 * row['util_change_mean'])
        + 0.10 * np.exp(-5.0 * row['positive_change_mean'])
        + 0.10 * np.exp(-row['rank_change_mean'])
    )


def choose_resolution_v5b(
    transition_df,
    resolutions,
    min_feature_gain,
    overscale_drop
):
    """
    Start from smallest scale and move upward.

    Stop before a transition if:
    - latent stability score is <= min_feature_gain, OR
    - score drops sharply vs previous transition.

    If no degradation detected, choose largest tested scale.
    """
    df = (
        transition_df
        .sort_values('res_small')
        .reset_index(drop=True)
        .copy()
    )

    df['latent_score'] = df.apply(
        latent_marginal_score,
        axis=1
    )

    selected = int(
        min(resolutions)
    )
    prev_score = None
    reason = 'reached_max_without_latent_degradation'

    for _, row in df.iterrows():
        rs = int(row['res_small'])
        rl = int(row['res_large'])
        score = float(
            row['latent_score']
        )

        if score <= min_feature_gain:
            selected = rs
            reason = (
                f'low_latent_score_at_{rs}_to_{rl}'
            )
            break

        if (
            prev_score is not None
            and (prev_score - score) >= overscale_drop
        ):
            selected = rs
            reason = (
                f'latent_overscale_drop_at_{rs}_to_{rl}'
            )
            break

        selected = rl
        prev_score = score

    return selected, reason, df


def main():
    args = parse_args()

    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )
    print('[DEVICE]', device)

    model, cfg = load_model(
        args.source,
        args.target,
        device
    )

    collector = FeatureCollector(
        model,
        ENCODER_BLOCKS
    )

    img_dir = os.path.join(
        'inputs',
        args.target,
        'train',
        'images'
    )

    paths = sorted(
        glob(
            os.path.join(
                img_dir,
                '*' + cfg['img_ext']
            )
        )
    )

    if not paths:
        raise RuntimeError(img_dir)

    rng = np.random.RandomState(
        args.seed
    )

    if (
        args.n_images > 0
        and args.n_images < len(paths)
    ):
        idx = rng.choice(
            len(paths),
            size=args.n_images,
            replace=False
        )
        paths = [
            paths[i]
            for i in sorted(idx)
        ]

    resolutions = sorted(
        args.resolutions
    )

    canonical = (
        512
        if 512 in resolutions
        else resolutions[
            len(resolutions)//2
        ]
    )

    print(
        f'[DATA] {args.source}->{args.target} '
        f'n={len(paths)}'
    )
    print(
        '[FEATURE BLOCKS]',
        ENCODER_BLOCKS
    )

    # -------------------------------------------
    # A. V3 Appearance Recovery
    # -------------------------------------------
    recovery_records = []
    raw_images = []

    for i, path in enumerate(paths):
        img = cv2.imread(
            path,
            cv2.IMREAD_COLOR
        )
        if img is None:
            raise RuntimeError(path)

        raw_images.append(img)

        fr = frangi_map(
            img,
            args.diag_size
        )

        raw = img
        clh = apply_clahe_lab(
            img
        )

        raw_f = cv2.flip(
            raw,
            1
        )
        clh_f = cv2.flip(
            clh,
            1
        )

        p_raw = resize01(
            predict(
                model,
                raw,
                canonical,
                device
            ),
            args.diag_size
        )

        p_clh = resize01(
            predict(
                model,
                clh,
                canonical,
                device
            ),
            args.diag_size
        )

        pf_raw = resize01(
            cv2.flip(
                predict(
                    model,
                    raw_f,
                    canonical,
                    device
                ),
                1
            ),
            args.diag_size
        )

        pf_clh = resize01(
            cv2.flip(
                predict(
                    model,
                    clh_f,
                    canonical,
                    device
                ),
                1
            ),
            args.diag_size
        )

        recovery_records.append(
            recovery_metrics(
                p_raw,
                p_clh,
                pf_raw,
                pf_clh,
                fr,
                args.threshold,
                args.recovery_delta,
                args.frangi_threshold
            )
        )

        print(
            f'[APP {i+1}/{len(paths)}] '
            f'{os.path.basename(path)}'
        )

    app = summarize(
        recovery_records
    )

    app['appearance_score'] = appearance_score(
        app
    )

    selected_clahe = bool(
        app['appearance_score'] > 0.0
    )

    # -------------------------------------------
    # B. Multi-level latent features
    # -------------------------------------------
    features_by_res = {
        r: []
        for r in resolutions
    }

    for i, img in enumerate(
        raw_images
    ):
        proc = (
            apply_clahe_lab(img)
            if selected_clahe
            else img
        )

        for r in resolutions:
            feats = collect_features(
                model,
                collector,
                proc,
                r,
                device,
                args.pool_size
            )

            # Ensure real forward blocks are active
            missing = [
                x for x in ENCODER_BLOCKS
                if x not in feats
            ]

            if missing:
                raise RuntimeError(
                    f'Feature hooks missing outputs: {missing}'
                )

            features_by_res[r].append(
                feats
            )

        print(
            f'[LATENT {i+1}/{len(raw_images)}] '
            f'all resolutions done'
        )

    # -------------------------------------------
    # C. Adjacent-scale latent transitions
    # -------------------------------------------
    transition_rows = []
    layer_rows = []

    for j in range(
        len(resolutions)-1
    ):
        rs = resolutions[j]
        rl = resolutions[j+1]

        recs = []

        for i in range(
            len(raw_images)
        ):
            m, per_layer = transition_feature_metrics(
                features_by_res[rs][i],
                features_by_res[rl][i],
                ENCODER_BLOCKS
            )

            recs.append(m)

            for lr in per_layer:
                layer_rows.append({
                    'image_index': i,
                    'res_small': rs,
                    'res_large': rl,
                    **lr
                })

        row = {
            'res_small': int(rs),
            'res_large': int(rl),
        }
        row.update(
            summarize(recs)
        )
        transition_rows.append(
            row
        )

    trans_df = pd.DataFrame(
        transition_rows
    )

    selected_res, scale_reason, ranked_df = choose_resolution_v5b(
        trans_df,
        resolutions,
        args.min_feature_gain,
        args.overscale_drop
    )

    layer_df = pd.DataFrame(
        layer_rows
    )

    # -------------------------------------------
    # D. Bootstrap stability
    # -------------------------------------------
    wins = {}
    B = max(
        0,
        int(args.bootstrap)
    )

    for _ in range(B):
        sample = rng.choice(
            len(raw_images),
            size=len(raw_images),
            replace=True
        ).tolist()

        # appearance bootstrap
        recs = [
            recovery_records[i]
            for i in sample
        ]

        a = summarize(
            recs
        )
        a['appearance_score'] = appearance_score(
            a
        )
        c = bool(
            a['appearance_score'] > 0.0
        )

        if c != selected_clahe:
            key = (
                'appearance_flip',
                -1
            )
            wins[key] = wins.get(
                key,
                0
            ) + 1
            continue

        rows = []

        for j in range(
            len(resolutions)-1
        ):
            rs = resolutions[j]
            rl = resolutions[j+1]
            rr = []

            for i in sample:
                m, _ = transition_feature_metrics(
                    features_by_res[rs][i],
                    features_by_res[rl][i],
                    ENCODER_BLOCKS
                )
                rr.append(m)

            row = {
                'res_small': int(rs),
                'res_large': int(rl),
            }
            row.update(
                summarize(rr)
            )
            rows.append(
                row
            )

        bdf = pd.DataFrame(
            rows
        )

        r, _, _ = choose_resolution_v5b(
            bdf,
            resolutions,
            args.min_feature_gain,
            args.overscale_drop
        )

        key = (
            c,
            r
        )
        wins[key] = wins.get(
            key,
            0
        ) + 1

    win_rate = (
        wins.get(
            (
                selected_clahe,
                selected_res
            ),
            0
        ) / B
        if B > 0
        else None
    )

    appearance_flip_rate = (
        wins.get(
            (
                'appearance_flip',
                -1
            ),
            0
        ) / B
        if B > 0
        else None
    )

    collector.close()

    os.makedirs(
        os.path.dirname(
            args.out_prefix
        ) or '.',
        exist_ok=True
    )

    pd.DataFrame(
        [app]
    ).to_csv(
        args.out_prefix
        + '_appearance.csv',
        index=False
    )

    ranked_df.to_csv(
        args.out_prefix
        + '_latent_transitions.csv',
        index=False
    )

    layer_df.to_csv(
        args.out_prefix
        + '_latent_layers.csv',
        index=False
    )

    decision = {
        'source': args.source,
        'target': args.target,
        'selector_version': (
            'V5b_V3appearance_multilevel_latent_scale_consistency'
        ),
        'resolution': int(
            selected_res
        ),
        'clahe': bool(
            selected_clahe
        ),
        'appearance_score': float(
            app['appearance_score']
        ),
        'scale_reason': scale_reason,
        'bootstrap_win_rate': win_rate,
        'appearance_flip_rate': appearance_flip_rate,
        'n_images': len(
            raw_images
        ),
        'feature_blocks': ENCODER_BLOCKS,
        'pool_size': int(
            args.pool_size
        ),
        'uses_target_labels': False,
        'uses_source_images_or_masks': False,
        'uses_batchnorm_stats': False,
    }

    with open(
        args.out_prefix
        + '_decision.json',
        'w'
    ) as f:
        json.dump(
            decision,
            f,
            indent=2
        )

    print(
        '\n=== APPEARANCE RECOVERY (V3) ==='
    )
    for k, v in app.items():
        print(
            f'{k}: {v}'
        )

    print(
        '\n=== V5b LATENT TRANSITIONS ==='
    )
    print(
        ranked_df[
            [
                'res_small',
                'res_large',
                'latent_cosine_mean',
                'latent_cosine_min_mean',
                'energy_change_mean',
                'util_change_mean',
                'positive_change_mean',
                'rank_change_mean',
                'latent_score',
            ]
        ].to_string(
            index=False
        )
    )

    print(
        '\n=== V5b DECISION ==='
    )
    print(
        json.dumps(
            decision,
            indent=2
        )
    )


if __name__ == '__main__':
    main()
