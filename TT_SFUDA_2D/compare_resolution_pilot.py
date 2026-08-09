"""
compare_resolution_pilot.py
==============================
Doc results_resolution1024_pilot.csv (ghi boi tt_sfuda_2d_dualema.py qua
run_resolution_pilot.sh), tinh mean/std cot 'adapted_dice', so sanh THANG
voi 2 con so tham chieu - KHONG can tu doc log bang mat.

Cach dung:
    python run.py compare_resolution_pilot.py
    python run.py compare_resolution_pilot.py --csv results_resolution1024_pilot.csv
"""
import argparse
import csv

REFERENCE = {
    'TT-SFUDA goc (paper, CHASE->HRF)': (58.25, 1.06),
    'Dual-EMA+rollback 512px (da kiem chung, n=7)': (58.20, 0.09),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results_resolution1024_pilot.csv')
    return parser.parse_args()


def main():
    args = parse_args()

    dice_values = []
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dice_values.append(float(row['adapted_dice']) * 100)  # ve thang % giong tham chieu
            except (ValueError, KeyError):
                continue

    if not dice_values:
        print(f"KHONG doc duoc gia tri Dice nao tu {args.csv} - kiem tra lai file co dung khong.")
        return

    n = len(dice_values)
    mean = sum(dice_values) / n
    std = (sum((x - mean) ** 2 for x in dice_values) / n) ** 0.5 if n > 1 else 0.0

    print(f"Doc duoc {n} lan chay tu {args.csv}: {[f'{v:.2f}' for v in dice_values]}")
    print(f"\nPilot 1024px: {mean:.2f} +/- {std:.2f}  (n={n})")
    print("-" * 60)
    for name, (ref_mean, ref_std) in REFERENCE.items():
        diff = mean - ref_mean
        print(f"{name}: {ref_mean:.2f} +/- {ref_std:.2f}   "
              f"-> chenh lech: {'+' if diff >= 0 else ''}{diff:.2f}")

    print("-" * 60)
    best_ref_mean, best_ref_std = REFERENCE['Dual-EMA+rollback 512px (da kiem chung, n=7)']
    if n < 3:
        print(f"[!] CANH BAO: chi co {n} lan chay - CHUA DU de ket luan chac chan (can it nhat 3, "
              f"ly tuong 7 nhu quy uoc du an). Doc ket qua nay CHI de tham khao nhanh.")
    elif mean > best_ref_mean + 2 * best_ref_std:
        print(f"=> TIN HIEU MANH: vuot ro rang so voi 512px (chenh lon hon 2 lan do lech chuan da "
              f"biet cua cau hinh tot nhat truoc do). Dang mo rong len 7 seed + 3 domain shift con lai.")
    elif mean > best_ref_mean:
        print(f"=> Cao hon nhung chua chac chan (chenh nam trong bien do co the la nhieu ngau nhien "
              f"giua cac seed). Chay them seed truoc khi ket luan.")
    else:
        print(f"=> KHONG vuot 512px trong pilot nay. Hieu ung resolution zero-shot co the khong "
              f"chuyen thanh loi ich sau khi train dai - can xem xet ky (co the model overfit "
              f"nhanh hon o resolution cao voi dataset nho) truoc khi bo huong nay.")


if __name__ == '__main__':
    main()
