"""
sequential_scale_then_appearance.py
======================================
Kiem tra quy tac 2 BUOC TUAN TU thay vi 2 selector HOAN TOAN doc lap:

    Buoc 1: r* = argmax_r S_scale(r)          (marginalize qua CLAHE - DA XAC NHAN tot, Spearman 0.95)
    Buoc 2: clahe* = argmax_{a in {T,F}} CMR(r*, a)   (TAI DUNG r* vua chon, KHONG marginalize nua)

Khac voi appearance selector hoan toan doc lap (S_appearance marginalize qua
CA 4 resolution), buoc 2 o day CHI nhin CMR tai dung 1 diem r* da chon o
buoc 1 - kiem tra gia thuyet "CLAHE giup trung binh nhung khong giup tai
diem resolution toi uu" co dung khong.

Dung lai NGUYEN file selector_v6_cmr_audit/{source}_to_{target}.csv da co
(8 dong: 4 resolution x 2 clahe, co san ca cmr_nhd va dice_mean) - khong
can du lieu moi, khong can GPU.

Cach dung (chay trong TT_SFUDA_2D/, sau khi da co selector_v6_cmr_audit/):
    python run.py sequential_scale_then_appearance.py
"""
import os

import pandas as pd


SHIFTS = [
    ('chase_unet', 'hrf'),
    ('chase_unet', 'rite'),
    ('hrf_unet', 'chase'),
    ('hrf_unet', 'rite'),
]

AUDIT_DIR = 'selector_v6_cmr_audit'


def main():
    rows = []

    for source, target in SHIFTS:
        path = os.path.join(AUDIT_DIR, f'{source}_to_{target}.csv')
        if not os.path.exists(path):
            print(f"[THIEU] {path}")
            continue

        df = pd.read_csv(path)
        shift_name = f'{source}->{target}'

        # Buoc 1: S_scale(r) = trung binh cmr_nhd qua {True, False} cho tung resolution
        s_scale = df.groupby('resolution')['cmr_nhd'].mean()
        r_star = int(s_scale.idxmax())

        # Buoc 2: TAI DUNG r_star, so cmr_nhd giua clahe True/False (KHONG marginalize)
        sub = df[df['resolution'] == r_star]
        clahe_star = bool(sub.loc[sub['cmr_nhd'].idxmax(), 'clahe'])

        # Dice THAT tai dung diem (r_star, clahe_star) da chon
        selected_row = df[(df['resolution'] == r_star) & (df['clahe'] == clahe_star)]
        selected_dice = float(selected_row['dice_mean'].iloc[0]) if len(selected_row) else None

        # Oracle THAT (Dice cao nhat trong ca 8 config, doi chieu hau kiem)
        oracle_row = df.loc[df['dice_mean'].idxmax()]
        oracle_res = int(oracle_row['resolution'])
        oracle_clahe = bool(oracle_row['clahe'])
        oracle_dice = float(oracle_row['dice_mean'])

        regret = oracle_dice - selected_dice if selected_dice is not None else None

        rows.append({
            'shift': shift_name,
            'sequential_selected': f'{r_star}/{"clahe" if clahe_star else "raw"}',
            'sequential_dice': selected_dice,
            'oracle': f'{oracle_res}/{"clahe" if oracle_clahe else "raw"}',
            'oracle_dice': oracle_dice,
            'regret': regret,
            'exact_match': (r_star == oracle_res) and (clahe_star == oracle_clahe),
        })

        print(f"\n{shift_name}")
        print(f"  Buoc 1 - S_scale chon r* = {r_star}")
        print(f"  Buoc 2 - tai r*={r_star}, CMR chon clahe = {clahe_star}")
        print(f"  Dice tai lua chon: {selected_dice}")
        print(f"  Oracle that: {oracle_res}/{'clahe' if oracle_clahe else 'raw'} (Dice={oracle_dice:.4f})")
        print(f"  Regret: {regret:.4f}" if regret is not None else "  Regret: N/A")

    if not rows:
        print("\nKhong doc duoc shift nao.")
        return

    result_df = pd.DataFrame(rows)
    print(f"\n{'='*70}\nTONG KET\n{'='*70}")
    print(result_df.to_string(index=False))
    print(f"\nMean regret (quy tac TUAN TU): {result_df['regret'].mean():.4f}")
    print(f"So khop tuyet doi: {result_df['exact_match'].sum()}/4")

    result_df.to_csv('sequential_scale_then_appearance_result.csv', index=False)
    print("\nDa luu sequential_scale_then_appearance_result.csv")


if __name__ == '__main__':
    main()
