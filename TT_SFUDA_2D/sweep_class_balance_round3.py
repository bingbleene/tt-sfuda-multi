"""
sweep_class_balance_round3.py
================================
fixed_fg van tang don dieu den 6.0, chua thay tran. Round nay do tiep len
8, 10, 12, 15 tren CA 4 domain shift (khong chi 2 cai kho) - vi round 2 da
xac nhan fixed_fg giup ca 4, can biet trong so toi uu co giong nhau giua
cac domain shift hay khac nhau.

CANH BAO can theo doi: khi fg_weight qua cao, model co the bat dau DU
DOAN QUA TAY (over-segment, false positive tran lan) - luc do Dice se
NGUNG tang roi GIAM XUONG (khong tang mai). Dau hieu nhan biet qua log:
iou giam dan du loss van giam.
"""
import subprocess

PAIRS = [('chase_unet', 'rite'), ('hrf_unet', 'rite'),
         ('chase_unet', 'hrf'), ('hrf_unet', 'chase')]
WEIGHTS = [8.0, 10.0, 12.0, 15.0]


def run_one(source, target, fg_weight):
    stage1_ckpt = f"cache/{source}_to_{target}_stage1_seed42.pth"
    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', source, '--target', target,
           '--topology', 'parallel', '--slow_keep_rate', '0.995',
           '--stage1_ckpt', stage1_ckpt, '--seed', '42',
           '--class_balance', '--class_balance_strategy', 'fixed_fg',
           '--fixed_fg_weight', str(fg_weight),
           '--results_csv', 'results_class_balance_round3.csv']

    print("=" * 70)
    print(' '.join(cmd))
    print("=" * 70)
    result = subprocess.run(cmd, capture_output=True, text=True)
    tail_lines = [l for l in result.stdout.splitlines()
                  if 'Adapted target model' in l or 'Da ghi ket qua' in l]
    for l in tail_lines:
        print(l)
    if result.returncode != 0:
        print("!!! LOI:")
        print(result.stderr[-1500:])


if __name__ == '__main__':
    total = len(PAIRS) * len(WEIGHTS)
    i = 0
    for source, target in PAIRS:
        for w in WEIGHTS:
            i += 1
            print(f"\n### Lan {i}/{total} ###")
            run_one(source, target, w)

    print(f"\nXong ({total} lan). Xem results_class_balance_round3.csv")
