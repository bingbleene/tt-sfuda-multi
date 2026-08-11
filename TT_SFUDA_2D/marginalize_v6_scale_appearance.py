"""
marginalize_v6_scale_appearance.py  (v2 - dung dung file/cot da xac nhan
tu chinh notebook V6 goc, khong con doan mo)
================================================================================
Doc lai file selector_v6_cmr_audit/{source}_to_{target}.csv DA CO SAN (audit_cmr_v6.py
da tu gop san CMR ranking + dice_mean vao 1 bang), tinh:

    S_scale(r)      = trung binh cmr_nhd qua {raw, clahe} cho tung resolution
    S_appearance(a) = trung binh cmr_nhd qua 4 resolution cho tung clahe

roi doi chieu voi dice_mean da marginalize tuong ung - kiem tra gia thuyet
"CMR chon dung resolution (3/4 shift) nhung CLAHE bi bias" bang cach xem
S_scale co tuong quan SACH HON voi Dice khi tach khoi anh huong CLAHE hay khong.

Cach dung (chay trong TT_SFUDA_2D/, sau khi da co selector_v6_cmr_audit/):
    python run.py marginalize_v6_scale_appearance.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SHIFTS = [
    ('chase_unet', 'hrf'),
    ('chase_unet', 'rite'),
    ('hrf_unet', 'chase'),
    ('hrf_unet', 'rite'),
]

AUDIT_DIR = 'selector_v6_cmr_audit'


def marginalize_one_shift(df):
    """df: bang da gop san (resolution, clahe, dice_mean, cmr_nhd, ...) tu
    audit_cmr_v6.py - dung THANG, khong can merge gi them."""
    scale_table = df.groupby('resolution').agg(
        S_scale=('cmr_nhd', 'mean'),
        dice_marginalized_over_clahe=('dice_mean', 'mean'),
    ).reset_index()

    appearance_table = df.groupby('clahe').agg(
        S_appearance=('cmr_nhd', 'mean'),
        dice_marginalized_over_resolution=('dice_mean', 'mean'),
    ).reset_index()

    scale_corr = None
    if len(scale_table) >= 3:
        rho, _ = spearmanr(scale_table['S_scale'], scale_table['dice_marginalized_over_clahe'])
        scale_corr = float(rho)

    return scale_table, appearance_table, scale_corr


def main():
    all_scale, all_appearance = [], []
    scale_corrs = {}

    for source, target in SHIFTS:
        path = os.path.join(AUDIT_DIR, f'{source}_to_{target}.csv')
        if not os.path.exists(path):
            print(f"[THIEU] {path}")
            continue

        df = pd.read_csv(path)
        shift_name = f'{source}->{target}'

        scale_table, appearance_table, scale_corr = marginalize_one_shift(df)
        scale_table['shift'] = shift_name
        appearance_table['shift'] = shift_name
        all_scale.append(scale_table)
        all_appearance.append(appearance_table)
        scale_corrs[shift_name] = scale_corr

        print(f"\n{'='*70}\n{shift_name}")
        print("\nS_scale(r) vs Dice (marginalized qua CLAHE):")
        print(scale_table.to_string(index=False))
        print(f"Spearman(S_scale, Dice_marginalized) = {scale_corr}")

        print("\nS_appearance(clahe) vs Dice (marginalized qua resolution):")
        print(appearance_table.to_string(index=False))
        app_sorted = appearance_table.sort_values('clahe')
        if len(app_sorted) == 2:
            s_diff = app_sorted['S_appearance'].iloc[1] - app_sorted['S_appearance'].iloc[0]
            d_diff = app_sorted['dice_marginalized_over_resolution'].iloc[1] - app_sorted['dice_marginalized_over_resolution'].iloc[0]
            if (s_diff > 0) != (d_diff > 0):
                print(f"  [CANH BAO] S_appearance va Dice di NGUOC CHIEU nhau khi bat CLAHE "
                      f"(S thay doi {s_diff:+.4f}, Dice thay doi {d_diff:+.4f}) - dung bang chung "
                      f"cho gia thuyet CLAHE bias.")

    if not all_scale:
        print("\nKhong doc duoc shift nao - kiem tra lai thu muc selector_v6_cmr_audit/ con khong.")
        return

    combined_scale = pd.concat(all_scale, ignore_index=True)
    combined_appearance = pd.concat(all_appearance, ignore_index=True)
    combined_scale.to_csv('marginalize_v6_scale.csv', index=False)
    combined_appearance.to_csv('marginalize_v6_appearance.csv', index=False)

    print(f"\n{'='*70}\nTONG KET")
    print(f"Spearman S_scale vs Dice-marginalized theo tung shift: {scale_corrs}")
    valid = [v for v in scale_corrs.values() if v is not None]
    if valid:
        print(f"Trung binh: {np.mean(valid):.4f}  (doi chieu voi Spearman 8-config goc cua V6: 0.875)")

    print("\nDa luu marginalize_v6_scale.csv va marginalize_v6_appearance.csv")


if __name__ == '__main__':
    main()
