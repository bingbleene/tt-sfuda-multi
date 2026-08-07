"""
sweep_early_stop_check.py
============================
Kiem tra co che unsupervised early-stop tren CA 4 domain shift, voi tran
20 epoch. Muc tieu: xac nhan co che KHONG bao dong gia (dung qua som) tren
cac domain shift von khong sup, va van dung dung luc tren cac domain shift
co sup (nhu chase->hrf da xac nhan).

So sanh voi mean 3 seed cua cau hinh 'full' (5-10 epoch goc) da co tu truoc,
xem early-stop (tran 20 epoch) co cho ket qua tuong duong hoac tot hon khong.
"""
import subprocess

PAIRS = [('chase_unet', 'hrf'), ('chase_unet', 'rite'),
         ('hrf_unet', 'chase'), ('hrf_unet', 'rite')]


def run_one(source, target):
    stage1_ckpt = f"cache/{source}_to_{target}_stage1_seed42.pth"
    cmd = ['python', 'run.py', 'tt_sfuda_2d_dualema.py',
           '--source', source, '--target', target,
           '--topology', 'parallel', '--slow_keep_rate', '0.995',
           '--class_balance', '--class_balance_strategy', 'fixed_fg', '--fixed_fg_weight', '6.0',
           '--stage2_epochs', '20', '--early_stop_unsupervised',
           '--stage1_ckpt', stage1_ckpt, '--seed', '42',
           '--results_csv', 'results_early_stop_check.csv']

    print("=" * 70)
    print(f"{source} -> {target}")
    print("=" * 70)
    result = subprocess.run(cmd, capture_output=True, text=True)
    tail_lines = [l for l in result.stdout.splitlines()
                  if 'EarlyStop-KGS] DUNG' in l or 'Adapted target model' in l or 'Da ghi ket qua' in l]
    for l in tail_lines:
        print(l)
    if not any('DUNG SOM' in l for l in tail_lines):
        print(">>> KHONG kich hoat dung som - chay het 20 epoch <<<")
    if result.returncode != 0:
        print("!!! LOI:")
        print(result.stderr[-1500:])


if __name__ == '__main__':
    for source, target in PAIRS:
        run_one(source, target)
    print("\nXong. Xem results_early_stop_check.csv - cot stage2_epochs_actual "
          "cho biet dung som o epoch nao (neu co).")
