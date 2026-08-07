"""
sweep_class_balance.py
========================
So sanh 4 chien luoc class-balance tren nen mean/kr=0.995 (cau hinh tot
nhat hien tai) - CUNG dung cache Stage I (seed=42) da co san tu truoc,
dam bao so sanh cong bang.

Yeu cau: da co san cache/chase_unet_to_rite_stage1_seed42.pth va
cache/hrf_unet_to_rite_stage1_seed42.pth (tu sweep_dualema_round3.py hoac
cac lan chay truoc). Neu chua co, script se tu train Stage I moi (van dung
nhung se cham hon va tao cache moi).
"""
import subprocess

PAIRS = [('chase_unet', 'rite'), ('hrf_unet', 'rite')]

CONFIGS = [
    dict(name='no_calibration', class_balance=False),
    dict(name='cbmt', class_balance=True, strategy='cbmt'),
    dict(name='symmetric', class_balance=True, strategy='symmetric'),
    dict(name='fixed_fg_1.5', class_balance=True, strategy='fixed_fg', fixed_fg_weight=1.5),
    dict(name='fixed_fg_2.0', class_balance=True, strategy='fixed_fg', fixed_fg_weight=2.0),
    dict(name='fixed_fg_3.0', class_balance=True, strategy='fixed_fg', fixed_fg_weight=3.0),
]


def run_one(source, target, cfg):
    stage1_ckpt = f"cache/{source}_to_{target}_stage1_seed42.pth"
    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', source, '--target', target,
           '--topology', 'parallel', '--slow_keep_rate', '0.995',
           '--stage1_ckpt', stage1_ckpt, '--seed', '42',
           '--results_csv', 'results_class_balance.csv']
    if cfg['class_balance']:
        cmd += ['--class_balance', '--class_balance_strategy', cfg['strategy']]
        if 'fixed_fg_weight' in cfg:
            cmd += ['--fixed_fg_weight', str(cfg['fixed_fg_weight'])]

    print("=" * 70)
    print(f"[{cfg['name']}] {source}->{target}:", ' '.join(cmd))
    print("=" * 70)
    result = subprocess.run(cmd, capture_output=True, text=True)
    tail_lines = [l for l in result.stdout.splitlines()
                  if 'Adapted target model' in l or 'Da ghi ket qua' in l or '[CACHE]' in l]
    for l in tail_lines:
        print(l)
    if result.returncode != 0:
        print("!!! LOI:")
        print(result.stderr[-1500:])


if __name__ == '__main__':
    total = len(PAIRS) * len(CONFIGS)
    i = 0
    for source, target in PAIRS:
        for cfg in CONFIGS:
            i += 1
            print(f"\n### Lan {i}/{total} ###")
            run_one(source, target, cfg)

    print(f"\nXong ({total} lan). Xem results_class_balance.csv")
