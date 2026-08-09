"""
compare_resolution_pilot.py
==============================
Doc results_*.csv, tinh mean/std cot 'adapted_dice', so sanh THANG voi (1) so
lieu paper goc cho DUNG domain shift dang xet, va (2) baseline 512px cua
CHINH BAN (doc dong tu results_control_512_check.csv trong CUNG lan chay -
KHONG hardcode, vi moi domain shift co baseline rieng, chi CHASE->HRF moi
co so "58.20+-0.09" da kiem chung ky voi n=7 tu truoc).

Cach dung:
    python run.py compare_resolution_pilot.py --domain_shift chase_hrf
    python run.py compare_resolution_pilot.py --domain_shift chase_rite \\
        --csv results_768_clean.csv --control_csv results_control_512_check.csv
"""
import argparse
import csv

# So lieu THAT tu Table 1 paper goc (TT-SFUDA, arXiv:2203.15792)
PAPER_REFERENCE = {
    'chase_hrf': ('CHASE->HRF', 58.25, 1.06),
    'chase_rite': ('CHASE->RITE', 52.63, 1.36),
    'hrf_chase': ('HRF->CHASE', 64.95, 0.89),
    'hrf_rite': ('HRF->RITE', 58.37, 1.03),
}

# Rieng CHASE->HRF da co ket qua tu cua chinh du an, kiem chung ky (n=7,
# Dual-EMA+rollback tai 512px) - cac domain shift khac CHUA co so nay, se
# dung ket qua control_csv (chay trong CUNG lan nay) thay the.
OWN_512_REFERENCE = {
    'chase_hrf': ('Dual-EMA+rollback 512px (da kiem chung, n=7)', 58.20, 0.09),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results_resolution1024_pilot.csv',
                         help='File CSV chua ket qua o resolution MOI can danh gia.')
    parser.add_argument('--control_csv', default=None,
                         help='File CSV control test o 512px (neu co) - dung lam baseline '
                              'rieng cho domain shift nay khi chua co so kiem chung san.')
    parser.add_argument('--domain_shift', default='chase_hrf',
                         choices=list(PAPER_REFERENCE.keys()),
                         help='Domain shift dang danh gia - quyet dinh so lieu paper tham chieu dung.')
    return parser.parse_args()


def read_dice_values(csv_path):
    values = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    values.append(float(row['adapted_dice']) * 100)
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        return None
    return values


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    std = (sum((x - mean) ** 2 for x in values) / n) ** 0.5 if n > 1 else 0.0
    return mean, std, n


def main():
    args = parse_args()

    dice_values = read_dice_values(args.csv)
    if not dice_values:
        print(f"KHONG doc duoc gia tri Dice nao tu {args.csv} - kiem tra lai file co dung khong.")
        return

    mean, std, n = mean_std(dice_values)
    print(f"Doc duoc {n} lan chay tu {args.csv}: {[f'{v:.2f}' for v in dice_values]}")
    print(f"\nKet qua danh gia: {mean:.2f} +/- {std:.2f}  (n={n})")
    print("-" * 60)

    # Tham chieu 1: paper goc, DUNG domain shift
    label, ref_mean, ref_std = PAPER_REFERENCE[args.domain_shift]
    diff = mean - ref_mean
    print(f"Paper goc ({label}): {ref_mean:.2f} +/- {ref_std:.2f}   "
          f"-> chenh lech: {'+' if diff >= 0 else ''}{diff:.2f}")

    # Tham chieu 2: hoac so da kiem chung san (chi CHASE->HRF), hoac tu control_csv
    best_ref_mean, best_ref_std = ref_mean, ref_std  # mac dinh: dung paper lam moc so sanh "vuot ro"
    if args.domain_shift in OWN_512_REFERENCE:
        label2, ref2_mean, ref2_std = OWN_512_REFERENCE[args.domain_shift]
        diff2 = mean - ref2_mean
        print(f"{label2}: {ref2_mean:.2f} +/- {ref2_std:.2f}   "
              f"-> chenh lech: {'+' if diff2 >= 0 else ''}{diff2:.2f}")
        best_ref_mean, best_ref_std = ref2_mean, ref2_std
    elif args.control_csv:
        control_values = read_dice_values(args.control_csv)
        if control_values:
            c_mean, c_std, c_n = mean_std(control_values)
            diff2 = mean - c_mean
            print(f"Control test 512px (chinh ban, n={c_n}): {c_mean:.2f} +/- {c_std:.2f}   "
                  f"-> chenh lech: {'+' if diff2 >= 0 else ''}{diff2:.2f}")
            best_ref_mean, best_ref_std = c_mean, c_std
        else:
            print(f"[!] Khong doc duoc control_csv ({args.control_csv}) - chi so sanh voi paper.")
    else:
        print("[!] Chua co --control_csv va domain shift nay chua co so kiem chung san - "
              "chi so sanh voi paper. Chay control test 512px truoc de co doi chieu day du.")

    print("-" * 60)
    if n < 3:
        print(f"[!] CANH BAO: chi co {n} lan chay - CHUA DU de ket luan chac chan (can it nhat 3, "
              f"ly tuong 7 nhu quy uoc du an). Doc ket qua nay CHI de tham khao nhanh.")
    elif mean > best_ref_mean + 2 * best_ref_std:
        print(f"=> TIN HIEU MANH: vuot ro rang so voi doi chieu (chenh lon hon 2 lan do lech chuan). "
              f"Dang mo rong danh gia day du (them seed / domain shift khac).")
    elif mean > best_ref_mean:
        print(f"=> Cao hon nhung chua chac chan (chenh nam trong bien do co the la nhieu ngau nhien). "
              f"Chay them seed truoc khi ket luan.")
    else:
        print(f"=> KHONG vuot doi chieu trong lan chay nay.")


if __name__ == '__main__':
    main()

