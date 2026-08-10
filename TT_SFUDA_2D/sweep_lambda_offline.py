"""
sweep_lambda_offline.py
==========================
Quet lambda DAY HON (0.0 -> 1.0, buoc 0.1) tu cac file *_transitions.csv
DA CO SAN (do unsupervised_selector_v5b.py da luu) - KHONG can chay lai
GPU/model, chi doc lai feature_utility da tinh san va thu nhieu lambda.

Dong thoi thu CACH CHON KHAC: thay vi "dung o transition dau tien fail",
thu "chon resolution co combined utility CAO NHAT toan dai" - de xem co
cho phep chon duoc 768 (dang bi bo qua o ca 2 cach chon nhi phan 512/1024
truoc do) hay khong.

Cach dung (chay TRONG TT_SFUDA_2D/, sau khi da co san selector_v5b_outputs/):
    python run.py sweep_lambda_offline.py \\
        --transitions_dir selector_v5b_outputs \\
        --audit_dir selector_v5b_audit
"""
import os
import argparse
import glob as globmod

import pandas as pd
from geometry_prior import combined_marginal_utility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--transitions_dir', default='selector_v5b_outputs')
    p.add_argument('--audit_dir', default='selector_v5b_audit')
    p.add_argument('--lambdas', type=float, nargs='+',
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument('--stop_drop_ratio', type=float, default=0.85)
    return p.parse_args()


SHIFTS = [
    ('chase_unet', 'hrf'),
    ('chase_unet', 'rite'),
    ('hrf_unet', 'chase'),
    ('hrf_unet', 'rite'),
]


def select_by_stop_rule(trans_df, col, resolutions, stop_drop_ratio):
    """Cach chon DA DUNG trong V5b that (khop chinh xac unsupervised_selector_v5b.py):
    dung o transition dau tien co combined utility < stop_drop_ratio * max_da_thay,
    lay res_small cua no."""
    max_seen = trans_df[col].iloc[0]
    selected_res = resolutions[0]
    for _, row in trans_df.iterrows():
        if row[col] < stop_drop_ratio * max_seen:
            selected_res = int(row['res_small'])
            break
        max_seen = max(max_seen, row[col])
        selected_res = int(row['res_large'])
    return selected_res


# LUU Y: ban dau co du dinh them 1 cach chon "global max" (trung binh utility
# cua 2 transition ke can moi resolution) de xem co chon duoc 768 khong.
# DA TU TEST TRUOC KHI GUI va phat hien no bi THIEN VI 2 DAU MUT (384/1024
# chi co 1 "hang xom" nen trung binh de cao hon bat cong bang, cho ra 384
# thay vi 768 trong test gia lap - dung loai loi endpoint-bias da tung
# canh bao truoc do). KHONG dua vao ban nay vi chua sua dung - can thiet ke
# lai (vd dung tich luy utility tu dau thay vi trung binh 2 hang xom) truoc
# khi dang tin cay. Chi dung select_by_stop_rule() (da kiem chung dung) o duoi.


def read_audit_dice(audit_path, resolution, clahe):
    df = pd.read_csv(audit_path)
    row = df[(df['resolution'] == resolution) & (df['clahe'] == clahe)]
    if len(row) == 0:
        return None
    return float(row.iloc[0]['dice_mean'])


def main():
    args = parse_args()

    all_rows = []
    for source, target in SHIFTS:
        trans_path = os.path.join(args.transitions_dir, f'{source}_to_{target}_transitions.csv')
        decision_path = os.path.join(args.transitions_dir, f'{source}_to_{target}_decision.json')
        audit_path = os.path.join(args.audit_dir, f'{source}_to_{target}_audit.csv')

        if not os.path.exists(trans_path):
            print(f"[THIEU] {trans_path}")
            continue

        trans_df = pd.read_csv(trans_path)
        resolutions = sorted(set(trans_df['res_small'].tolist() + trans_df['res_large'].tolist()))

        import json
        with open(decision_path) as f:
            decision = json.load(f)
        clahe = bool(decision['clahe'])

        oracle_df = pd.read_csv(audit_path)
        oracle_row = oracle_df.sort_values('dice_mean', ascending=False).iloc[0]
        oracle_dice = float(oracle_row['dice_mean'])

        for lam in args.lambdas:
            col = f'_tmp_combined_{lam}'
            trans_df[col] = trans_df.apply(
                lambda row: combined_marginal_utility(
                    row['feature_utility'], int(row['res_small']), int(row['res_large']),
                    target, source, lam),
                axis=1)

            res_stop = select_by_stop_rule(trans_df, col, resolutions, args.stop_drop_ratio)
            dice_stop = read_audit_dice(audit_path, res_stop, clahe)

            all_rows.append({
                'shift': f'{source}->{target}', 'lambda': lam,
                'res_stop_rule': res_stop,
                'dice_stop_rule': dice_stop,
                'regret_stop_rule': (oracle_dice - dice_stop) if dice_stop else None,
            })

    result_df = pd.DataFrame(all_rows)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 20)
    print(result_df.to_string(index=False))

    print("\n=== Regret trung binh theo lambda ===")
    print(result_df.groupby('lambda')['regret_stop_rule'].agg(['mean', 'max']).to_string())

    result_df.to_csv('sweep_lambda_offline_result.csv', index=False)
    print("\nDa luu sweep_lambda_offline_result.csv")


if __name__ == '__main__':
    main()
