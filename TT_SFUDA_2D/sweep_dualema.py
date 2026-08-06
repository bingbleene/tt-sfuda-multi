"""
sweep_dualema.py
=================
Chay TU DONG nhieu cau hinh Dual-EMA lien tiep, ghi tat ca ket qua vao
results_dualema.csv (append, khong ghi de - chay lai nhieu lan van tich luy).

Uu tien sweep tren 2 domain shift dang co van de (target=RITE), vi day la
noi can tim cau hinh khac phuc. Kem 1 domain shift "tot" (CHASE->HRF) de
kiem tra KHONG lam hong ket qua da tot san.

Chay tren Kaggle: dat file nay trong TT_SFUDA_2D/, chay:
    !python sweep_dualema.py
(mat khoang 15-25 phut cho toan bo sweep ben duoi, tuy GPU)
"""
import subprocess

RUNS = [
    # ---- Nhom 1: doi ensemble_mode, giu keep_rate mac dinh (0.99/0.999) ----
    dict(source='chase_unet', target='rite', topology='parallel', ensemble_mode='weighted', fast_weight=0.8),
    dict(source='chase_unet', target='rite', topology='parallel', ensemble_mode='confidence'),
    dict(source='hrf_unet', target='rite', topology='parallel', ensemble_mode='weighted', fast_weight=0.8),
    dict(source='hrf_unet', target='rite', topology='parallel', ensemble_mode='confidence'),

    # ---- Nhom 2: sweep slow_keep_rate (nhe hon 0.999), giu ensemble_mode=mean ----
    dict(source='chase_unet', target='rite', topology='parallel', slow_keep_rate=0.993),
    dict(source='chase_unet', target='rite', topology='parallel', slow_keep_rate=0.995),
    dict(source='chase_unet', target='rite', topology='parallel', slow_keep_rate=0.997),
    dict(source='hrf_unet', target='rite', topology='parallel', slow_keep_rate=0.993),
    dict(source='hrf_unet', target='rite', topology='parallel', slow_keep_rate=0.995),
    dict(source='hrf_unet', target='rite', topology='parallel', slow_keep_rate=0.997),

    # ---- Nhom 3: kiem tra khong lam hong CHASE->HRF (domain shift dang tot) ----
    dict(source='chase_unet', target='hrf', topology='parallel', ensemble_mode='weighted', fast_weight=0.8),
    dict(source='chase_unet', target='hrf', topology='parallel', ensemble_mode='confidence'),
]


def run_one(cfg):
    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', cfg['source'], '--target', cfg['target'],
           '--topology', cfg.get('topology', 'parallel'),
           '--ensemble_mode', cfg.get('ensemble_mode', 'mean')]
    if 'slow_keep_rate' in cfg:
        cmd += ['--slow_keep_rate', str(cfg['slow_keep_rate'])]
    if 'fast_weight' in cfg:
        cmd += ['--fast_weight', str(cfg['fast_weight'])]

    print("=" * 70)
    print("CHAY:", ' '.join(cmd))
    print("=" * 70)
    result = subprocess.run(cmd, capture_output=True, text=True)
    # chi in dong cuoi (ket qua Dice) de log gon, khong spam toan bo tqdm
    tail_lines = [l for l in result.stdout.splitlines() if 'Adapted target model' in l or 'Da ghi ket qua' in l]
    for l in tail_lines:
        print(l)
    if result.returncode != 0:
        print("!!! LOI, xem chi tiet:")
        print(result.stderr[-1500:])


if __name__ == '__main__':
    for i, cfg in enumerate(RUNS):
        print(f"\n### Lan {i+1}/{len(RUNS)} ###")
        run_one(cfg)

    print("\nXong toan bo sweep. Xem ket qua trong results_dualema.csv")