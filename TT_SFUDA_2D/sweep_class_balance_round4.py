"""
sweep_class_balance_round4.py
================================
Dien not du lieu con thieu: fg_weight = 4,5,6,7 tren CA 4 domain shift
(round truoc chi test day du tren 2 domain shift kho). Muc tieu CUOI CUNG:
chon 1 gia tri fg_weight DUY NHAT, dung chung cho ca 4 domain shift (khong
cherry-pick rieng tung cai) - dua tren TONG hoac TRUNG BINH Dice ca 4.
"""
import subprocess

PAIRS = [('chase_unet', 'rite'), ('hrf_unet', 'rite'),
         ('chase_unet', 'hrf'), ('hrf_unet', 'chase')]
WEIGHTS = [4.0, 5.0, 6.0, 7.0]

# danh dau nhung cap (source,target,weight) DA CO SAN tu round truoc,
# khong can chay lai (tiet kiem thoi gian)
ALREADY_HAVE = {
    ('chase_unet', 'rite', 4.0), ('chase_unet', 'rite', 5.0), ('chase_unet', 'rite', 6.0),
    ('hrf_unet', 'rite', 4.0), ('hrf_unet', 'rite', 5.0), ('hrf_unet', 'rite', 6.0),
}


def run_one(source, target, fg_weight):
    stage1_ckpt = f"cache/{source}_to_{target}_stage1_seed42.pth"
    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', source, '--target', target,
           '--topology', 'parallel', '--slow_keep_rate', '0.995',
           '--stage1_ckpt', stage1_ckpt, '--seed', '42',
           '--class_balance', '--class_balance_strategy', 'fixed_fg',
           '--fixed_fg_weight', str(fg_weight),
           '--results_csv', 'results_class_balance_round4.csv']

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
    todo = [(s, t, w) for (s, t) in PAIRS for w in WEIGHTS
            if (s, t, w) not in ALREADY_HAVE]
    print(f"Se chay {len(todo)} lan (da bo qua {len(PAIRS)*len(WEIGHTS)-len(todo)} lan da co san tu round truoc)")

    for i, (source, target, w) in enumerate(todo, 1):
        print(f"\n### Lan {i}/{len(todo)} ###")
        run_one(source, target, w)

    print(f"\nXong. Xem results_class_balance_round4.csv (nho GOP them ket qua fg=4,5,6 "
          f"cua chase_rite/hrf_rite tu round truoc khi tong hop bang cuoi cung).")
