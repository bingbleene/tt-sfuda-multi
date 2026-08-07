"""
run_fair_baseline.py
======================
Chay "1-teacher" (tuong duong toan hoc voi baseline goc cua tac gia) NHUNG
qua CUNG checkpoint Stage I da cache tu sweep_dualema_round3.py - de co
cot baseline THUC SU cong bang khi doi chieu voi cac cau hinh Dual-EMA.

PHAI chay sweep_dualema_round3.py TRUOC (de co san cache/*.pth), neu khong
se tu train Stage I moi (van dung nhung mat cong bang voi cac cot da co).
"""
import subprocess

PAIRS = [
    ('chase_unet', 'hrf'),
    ('chase_unet', 'rite'),
    ('hrf_unet', 'chase'),
    ('hrf_unet', 'rite'),
]


def run_one(source, target):
    stage1_ckpt = f"cache/{source}_to_{target}_stage1.pth"
    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', source, '--target', target,
           '--single_teacher',
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
    for source, target in PAIRS:
        run_one(source, target)
    print("\nXong. Cot 'adapted_dice' voi topology=parallel, ensemble_mode=mean, "
          "slow_keep_rate=None trong results_dualema.csv chinh la baseline CONG BANG.")
