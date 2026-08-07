"""
sweep_ablation_multiseed.py
==============================
Buoc xac nhan cuoi cung truoc khi dua vao bao cao. 2 muc tieu gop lam 1:

1. ABLATION: tach rieng dong gop cua tung thanh phan, khong chi bao cao
   con so "cong don":
     - baseline        : 1 teacher, khong class_balance (dung --single_teacher)
     - dual_ema_only    : parallel/kr=0.995, KHONG class_balance
     - class_balance_only: 1 teacher (--single_teacher), CO class_balance fg=6.0
     - full             : parallel/kr=0.995 + class_balance fg=6.0 (cau hinh de xuat)

2. MULTI-SEED: moi cau hinh chay voi 3 seed DOC LAP (42,43,44) - moi seed
   TU TRAIN LAI Stage I rieng (khong dung chung cache) - phan anh dung
   nhieu THUC TE cua toan bo pipeline (khong chi Stage II).

Tong: 4 domain shift x 4 cau hinh x 3 seed = 48 lan chay. Co the ngat giua
chung (Ctrl+C hoac dong notebook) va chay lai sau - CSV ghi APPEND, khong
mat du lieu da chay, chi can bo qua nhung dong da co san khi doc ket qua.

Uoc tinh thoi gian: ~40-50s/lan (RITE) den ~30s/lan (CHASE/HRF -> nho hon)
-> tong khoang 30-40 phut cho ca 48 lan.
"""
import subprocess

PAIRS = [('chase_unet', 'rite'), ('hrf_unet', 'rite'),
         ('chase_unet', 'hrf'), ('hrf_unet', 'chase')]
SEEDS = [42, 43, 44]

CONFIGS = [
    dict(name='baseline', single_teacher=True, class_balance=False),
    dict(name='dual_ema_only', single_teacher=False, class_balance=False),
    dict(name='class_balance_only', single_teacher=True, class_balance=True),
    dict(name='full', single_teacher=False, class_balance=True),
]


def run_one(source, target, seed, cfg):
    # MOI seed tu train Stage I RIENG (khong dung chung cache giua cac seed)
    stage1_ckpt = f"cache/{source}_to_{target}_stage1_seed{seed}.pth"
    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', source, '--target', target,
           '--seed', str(seed),
           '--stage1_ckpt', stage1_ckpt,
           '--results_csv', 'results_ablation_multiseed.csv']

    if cfg['single_teacher']:
        cmd += ['--single_teacher']
    else:
        cmd += ['--topology', 'parallel', '--slow_keep_rate', '0.995']

    if cfg['class_balance']:
        cmd += ['--class_balance', '--class_balance_strategy', 'fixed_fg', '--fixed_fg_weight', '6.0']

    print("=" * 70)
    print(f"[{cfg['name']}] seed={seed} {source}->{target}")
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
    total = len(PAIRS) * len(SEEDS) * len(CONFIGS)
    i = 0
    for source, target in PAIRS:
        for seed in SEEDS:
            for cfg in CONFIGS:
                i += 1
                print(f"\n### Lan {i}/{total} ###")
                run_one(source, target, seed, cfg)

    print(f"\nXong ({total} lan). Xem results_ablation_multiseed.csv")
    print("Tinh mean +/- std theo (source,target,config) truoc khi ket luan.")
