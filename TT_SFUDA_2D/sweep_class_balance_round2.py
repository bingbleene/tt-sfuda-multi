"""
sweep_class_balance_round2.py
================================
Dua tren phat hien round 1: fixed_fg co xu huong TANG DON DIEU theo he so
(1.5 -> 2.0 -> 3.0 deu tang), CHUA thay dau hieu cham tran. Round nay:
  (a) Mo rong he so len 4.0, 5.0, 6.0 tren 2 domain shift kho (target=RITE)
      de tim diem cham tran / bat dau giam.
  (b) REGRESSION CHECK: ap dung fixed_fg=3.0 (ung vien thang cuoc) len 2
      domain shift DANG TOT (CHASE->HRF, HRF->CHASE) - dam bao khong lam
      hong nhung gi da co.
"""
import subprocess

# ---- (a) do tran fixed_fg tren target=RITE ----
HARD_PAIRS = [('chase_unet', 'rite'), ('hrf_unet', 'rite')]
HIGHER_WEIGHTS = [4.0, 5.0, 6.0]

# ---- (b) regression check fixed_fg=3.0 tren domain shift dang tot ----
EASY_PAIRS = [('chase_unet', 'hrf'), ('hrf_unet', 'chase')]
BEST_WEIGHT_SO_FAR = 3.0


def run_one(source, target, class_balance, strategy=None, fg_weight=None):
    stage1_ckpt = f"cache/{source}_to_{target}_stage1_seed42.pth"
    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', source, '--target', target,
           '--topology', 'parallel', '--slow_keep_rate', '0.995',
           '--stage1_ckpt', stage1_ckpt, '--seed', '42',
           '--results_csv', 'results_class_balance_round2.csv']
    if class_balance:
        cmd += ['--class_balance', '--class_balance_strategy', strategy,
                '--fixed_fg_weight', str(fg_weight)]

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
    print("### PHAN (a): do tran fixed_fg tren target=RITE ###")
    for source, target in HARD_PAIRS:
        for w in HIGHER_WEIGHTS:
            run_one(source, target, class_balance=True, strategy='fixed_fg', fg_weight=w)

    print("\n### PHAN (b): regression check fixed_fg=3.0 tren domain shift dang tot ###")
    for source, target in EASY_PAIRS:
        run_one(source, target, class_balance=False)  # doi chung: khong hieu chinh
        run_one(source, target, class_balance=True, strategy='fixed_fg', fg_weight=BEST_WEIGHT_SO_FAR)

    print("\nXong. Xem results_class_balance_round2.csv")
