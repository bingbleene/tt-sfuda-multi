"""
check_resolution_gap.py
==========================
Y TUONG A (huong "think out of the box"): nghi ngo tran Dice ~58 CHUNG cho
MOI phuong phap (paper goc lan moi thu nghiem cua ban) khong phai do thuat
toan hoc, ma do BUOC TIEN XU LY - anh bi resize ve input_h x input_w co
dinh TRUOC KHI vao model. Neu anh goc (dac biet HRF, do phan giai rat cao)
bi thu nho manh, mach mau 1-3px co the co xuong DUOI 1 PIXEL - thong tin
mat VINH VIEN, khong thuat toan Stage I/II/rollback/ensemble nao cuu duoc.

Script nay CHI DO, KHONG SUA GI - in ra ty le thu nho thuc te va UOC TINH
do rong mach mau con lai (pixel) SAU resize, dua tren gia dinh mach mau
goc rong 1-3px o do phan giai native.

Cach dung (chay trong TT_SFUDA_2D/ tren Kaggle):
    python run.py check_resolution_gap.py
"""
import os
import yaml
from glob import glob

import cv2
import numpy as np

# Dataset -> 1 config bat ky co target = dataset do, chi de lay input_h/input_w
# (gia dinh giong nhau giua cac source cho cung 1 target - dung quy uoc da
# dung trong diagnose_gap_v2.py va validate_wavelet_prior.py).
DATASET_CONFIG_SOURCE = {
    'chase': ('hrf_unet', 'config_chase_dualema'),
    'hrf': ('chase_unet', 'config_hrf_dualema'),
    'rite': ('chase_unet', 'config_rite_dualema'),
}

# Do rong mach mau THAT o do phan giai NATIVE (pixel) - gia dinh tu mo ta
# bai toan cua ban ("mach mau 1-3 pixel"). Neu ban co so lieu chinh xac hon
# rieng cho tung dataset (CHASE/HRF/RITE co the khac nhau do thiet bi chup
# khac nhau), sua truc tiep dict nay.
NATIVE_VESSEL_WIDTH_PX = (1, 3)


def load_config_for_dataset(dataset_name):
    source, config_file = DATASET_CONFIG_SOURCE[dataset_name]
    with open(f'models/{source}/{config_file}.yml', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    print(f"{'Dataset':<8} {'Anh goc (HxW)':<18} {'Resize ve (HxW)':<18} "
          f"{'Ty le thu nho':<16} {'Do rong mach mau CON LAI (px)':<30}")
    print("-" * 95)

    for dataset_name in ['chase', 'hrf', 'rite']:
        config = load_config_for_dataset(dataset_name)
        input_h, input_w = config['input_h'], config['input_w']

        img_dir = os.path.join('inputs', dataset_name, 'train', 'images')
        img_paths = glob(os.path.join(img_dir, '*' + config['img_ext']))
        if not img_paths:
            print(f"{dataset_name:<8} [KHONG TIM THAY ANH TRONG {img_dir}]")
            continue

        # Doc vai anh dau de lay kich thuoc goc (thuong dong nhat trong 1
        # dataset, nhung doc nhieu anh de chac chan khong co ngoai le).
        native_sizes = []
        for p in img_paths[:5]:
            img = cv2.imread(p)
            if img is not None:
                native_sizes.append(img.shape[:2])  # (H, W)

        native_h = np.mean([s[0] for s in native_sizes])
        native_w = np.mean([s[1] for s in native_sizes])

        scale_h = input_h / native_h
        scale_w = input_w / native_w
        avg_scale = (scale_h + scale_w) / 2

        vmin, vmax = NATIVE_VESSEL_WIDTH_PX
        remaining_min = vmin * avg_scale
        remaining_max = vmax * avg_scale

        warning = ""
        if remaining_min < 1.0:
            warning = "  [!] MACH MAU MANH NHAT CO THE DA <1px SAU RESIZE - THONG TIN MAT VINH VIEN"
        elif remaining_min < 1.5:
            warning = "  [?] Gan nguong 1px - rui ro cao, nen kiem tra truc quan"

        print(f"{dataset_name:<8} {f'{native_h:.0f}x{native_w:.0f}':<18} "
              f"{f'{input_h}x{input_w}':<18} {f'{avg_scale:.3f}x':<16} "
              f"{f'{remaining_min:.2f} - {remaining_max:.2f} px':<30}{warning}")

    print("\n" + "=" * 95)
    print("DOC KET QUA: neu cot cuoi cho gia tri < 1.0px o BAT KY dataset nao (dac biet HRF,")
    print("do phan giai native rat cao), day la bang chung manh cho thay tran Dice ~58 hien tai")
    print("co the la GIOI HAN VAT LY tu buoc resize, KHONG PHAI gioi han thuat toan Stage I/II.")
    print("Neu dung: huong sua KHONG PHAI doi kien truc/loss nua, ma la training/inference theo")
    print("PATCH o do phan giai GAN NATIVE (crop cua so nho, KHONG resize toan anh xuong nho),")
    print("roi ghep lai - chuan trong xu ly anh y te cho cau truc mo/mach mau mong.")
    print("=" * 95)


if __name__ == '__main__':
    main()
