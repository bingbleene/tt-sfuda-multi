"""
sweep_dualema_round3.py
=========================
Chay lai TOAN BO cau hinh chinh (baseline moi topology/keep_rate da nghi la
tot) NHUNG voi --stage1_ckpt co dinh cho tung domain shift - dam bao khong
con nhieu tu viec train lai Stage I moi lan. Day la lan chay DUY NHAT co the
tin tuong de rut ket luan cuoi cung.

Lan chay DAU TIEN cho moi domain shift se train + luu Stage I; cac lan sau
CUNG domain shift se tu dong tai lai, nhanh hon nhieu.
"""
import subprocess

RUNS = [
    # CHASE -> HRF
    dict(source='chase_unet', target='hrf', topology='parallel', slow_keep_rate=0.999),
    dict(source='chase_unet', target='hrf', topology='cascaded', slow_keep_rate=0.999),
    dict(source='chase_unet', target='hrf', topology='parallel', slow_keep_rate=0.995),

    # CHASE -> RITE
    dict(source='chase_unet', target='rite', topology='parallel', slow_keep_rate=0.999),
    dict(source='chase_unet', target='rite', topology='cascaded', slow_keep_rate=0.999),
    dict(source='chase_unet', target='rite', topology='parallel', slow_keep_rate=0.995),

    # HRF -> CHASE
    dict(source='hrf_unet', target='chase', topology='parallel', slow_keep_rate=0.999),
    dict(source='hrf_unet', target='chase', topology='cascaded', slow_keep_rate=0.999),
    dict(source='hrf_unet', target='chase', topology='parallel', slow_keep_rate=0.995),

    # HRF -> RITE
    dict(source='hrf_unet', target='rite', topology='parallel', slow_keep_rate=0.999),
    dict(source='hrf_unet', target='rite', topology='cascaded', slow_keep_rate=0.999),
    dict(source='hrf_unet', target='rite', topology='parallel', slow_keep_rate=0.995),
]


def run_one(cfg):
    stage1_ckpt = f"cache/{cfg['source']}_to_{cfg['target']}_stage1.pth"
    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', cfg['source'], '--target', cfg['target'],
           '--topology', cfg['topology'],
           '--slow_keep_rate', str(cfg['slow_keep_rate']),
           '--stage1_ckpt', stage1_ckpt,
           '--seed', '42']

    print("=" * 70)
    print("CHAY:", ' '.join(cmd))
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
    for i, cfg in enumerate(RUNS):
        print(f"\n### Lan {i+1}/{len(RUNS)} ###")
        run_one(cfg)

    print("\nXong round 3 (co cache Stage I). Day la ket qua DANG TIN CAY nhat tu truoc den gio.")
