"""
sweep_dualema_round2.py
========================
Dua tren ket qua round 1: slow_keep_rate=0.995 la huong khac phuc dung cho
target=RITE. Round nay:
  (a) Kiem tra slow_keep_rate=0.995 co lam hong CHASE->HRF, HRF->CHASE
      (domain shift dang tot) khong - REGRESSION CHECK, quan trong nhat.
  (b) Thu ket hop slow_keep_rate=0.995 + ensemble_mode=confidence (2 huong
      tot nhat cong lai) tren ca 4 domain shift.
  (c) Hoan tat topology=cascaded voi keep_rate moi (chua test o round 1).
"""
import subprocess

RUNS = [
    # ---- (a) Regression check: kr=0.995 tren domain shift dang tot ----
    dict(source='chase_unet', target='hrf', topology='parallel', slow_keep_rate=0.995),
    dict(source='hrf_unet', target='chase', topology='parallel', slow_keep_rate=0.995),

    # ---- (b) Ket hop kr=0.995 + confidence, ca 4 domain shift ----
    dict(source='chase_unet', target='rite', topology='parallel', slow_keep_rate=0.995, ensemble_mode='confidence'),
    dict(source='hrf_unet', target='rite', topology='parallel', slow_keep_rate=0.995, ensemble_mode='confidence'),
    dict(source='chase_unet', target='hrf', topology='parallel', slow_keep_rate=0.995, ensemble_mode='confidence'),
    dict(source='hrf_unet', target='chase', topology='parallel', slow_keep_rate=0.995, ensemble_mode='confidence'),

    # ---- (c) cascaded voi keep_rate moi, ca 4 domain shift ----
    dict(source='chase_unet', target='rite', topology='cascaded', slow_keep_rate=0.995),
    dict(source='hrf_unet', target='rite', topology='cascaded', slow_keep_rate=0.995),
    dict(source='chase_unet', target='hrf', topology='cascaded', slow_keep_rate=0.995),
    dict(source='hrf_unet', target='chase', topology='cascaded', slow_keep_rate=0.995),
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

    print("\nXong round 2. Xem results_dualema.csv (da CONG DON voi round 1, khong bi mat).")