"""
Phan tich duong cong Dice theo epoch tu log console cua tt_sfuda_2d_control_cldice.py.

Muc dich: tinh 2 chi so moi (thay vi chi nhin Dice epoch cuoi):
  1. peak_dice       - Dice cao nhat dat duoc trong toan bo qua trinh Stage II
  2. final_dice      - Dice o epoch cuoi cung (dung de so sanh voi cach lam cu)
  3. drop            - peak_dice - final_dice (do "sup" tu dinh)
  4. collapsed       - True neu drop > COLLAPSE_THRESHOLD

Cach dung:
  1. Dan toan bo output console (nhieu lan chay lien tiep) vao file raw_logs.txt
  2. Chinh danh sach RUNS ben duoi cho khop voi thu tu cac lan chay thuc te
     (script khong tu doc duoc seed/lambda_cl tu log vi ban khong in ra dong
     "seed=..." - script tu doi chieu theo THU TU xuat hien trong file)
  3. python analyze_training_curves.py raw_logs.txt
"""

import re
import sys
import statistics as stats

COLLAPSE_THRESHOLD = 0.03  # nguong Dice sut tu dinh de tinh la "collapse"

# Khai bao THEO DUNG THU TU cac lan chay xuat hien trong file log dau vao.
# (seed, lambda_cl) - dieu chinh lai cho khop voi lich su chay thuc te cua ban.
RUNS = [
    (42, 0.0), (42, 1.0),
    (123, 0.0), (123, 1.0),
    (7, 0.0), (7, 1.0),
    (99, 0.0), (99, 1.0),
    (2024, 0.0), (2024, 1.0),
    (999, 0.0), (999, 1.0),
    (555, 0.0), (555, 1.0),
]

DICE_LINE = re.compile(
    r"Dice sau epoch (\d+):\s*([0-9.]+)"
)


def parse_runs(text: str):
    """Tach text thanh cac khoi, moi khoi tuong ung 1 lan chay (bat dau bang
    'Loading source trained model'), roi trich cac gia tri Dice theo epoch."""
    blocks = re.split(r"(?=Loading source trained model)", text)
    blocks = [b for b in blocks if b.strip()]
    runs = []
    for b in blocks:
        epochs = [float(v) for _, v in DICE_LINE.findall(b)]
        if epochs:
            runs.append(epochs)
    return runs


def main(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    runs = parse_runs(text)
    if len(runs) != len(RUNS):
        print(f"[CANH BAO] So lan chay tim duoc trong log ({len(runs)}) khac "
              f"so voi danh sach RUNS khai bao ({len(RUNS)}). Kiem tra lai "
              f"thu tu/khai bao RUNS truoc khi tin ket qua.")

    rows = []
    for (seed, lam), epochs in zip(RUNS, runs):
        peak = max(epochs)
        final = epochs[-1]
        drop = peak - final
        collapsed = drop > COLLAPSE_THRESHOLD
        rows.append(dict(seed=seed, lambda_cl=lam, peak=peak, final=final,
                          drop=drop, collapsed=collapsed, curve=epochs))

    print(f"{'seed':>6} {'lambda':>7} {'peak':>7} {'final':>7} {'drop':>7} {'collapsed':>10}")
    for r in rows:
        print(f"{r['seed']:>6} {r['lambda_cl']:>7.1f} {r['peak']:>7.4f} "
              f"{r['final']:>7.4f} {r['drop']:>7.4f} {str(r['collapsed']):>10}")

    print("\n--- Tong hop theo lambda_cl ---")
    for lam in sorted(set(r["lambda_cl"] for r in rows)):
        sub = [r for r in rows if r["lambda_cl"] == lam]
        drops = [r["drop"] for r in sub]
        n_collapsed = sum(r["collapsed"] for r in sub)
        peaks = [r["peak"] for r in sub]
        finals = [r["final"] for r in sub]
        print(f"lambda_cl={lam}: n={len(sub)} | "
              f"peak trung binh={stats.mean(peaks):.4f} | "
              f"final trung binh={stats.mean(finals):.4f} | "
              f"drop trung binh={stats.mean(drops):.4f} "
              f"(std={stats.pstdev(drops):.4f}) | "
              f"so lan collapse={n_collapsed}/{len(sub)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Cach dung: python analyze_training_curves.py raw_logs.txt")
        sys.exit(1)
    main(sys.argv[1])
