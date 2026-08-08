"""
regenerate_dualema_configs.py
==============================
Tao lai cac file config_{target}_dualema.yml bi mat khi restart phien Kaggle
(khong nam trong Kaggle Dataset goc, chi ton tai trong thu muc lam viec tam).

Giu NGUYEN moi khoa da co trong config.yml goc (Stage 0), CHI THEM 3 khoa
con thieu: stage1, stage2, teachers - dung dung gia tri da dung xuyen suot
du an tu truoc toi gio (khong doan mo):
  - lr: 0.0001 (rieng cho Stage I/II thich nghi - theo dung paper TT-SFUDA,
    khac voi lr=0.001 dung khi train Stage 0/source, xem "Implementation
    Details" trong paper goc)
  - stage1: 1 epoch (dung logic vong lap Stage I luon chi in 1 dong
    "train_loss X - train_iou X" khong co "epoch N/M", khop voi moi log
    tu truoc den gio trong du an)
  - stage2: 5 epoch (mac dinh khi khong truyen --stage2_epochs, khop voi
    "[INFO] Stage II se chay 5 epoch (tu config goc)" xuat hien trong MOI
    log truoc day khi khong ghi de)
  - teachers: fast (keep_rate=0.99) + slow (keep_rate=0.999 la gia tri MAC
    DINH trong yaml - --slow_keep_rate 0.995 se ghi de luc chay, dung nhu
    da lam trong hau het cac lenh truoc day)

Cach dung (chay trong thu muc TT_SFUDA_2D/ tren Kaggle):
    python regenerate_dualema_configs.py
"""
import os
import yaml
import copy

SOURCES = ['chase_unet', 'hrf_unet', 'rite_unet']
TARGETS = {
    'chase_unet': ['hrf', 'rite'],
    'hrf_unet':   ['chase', 'rite'],
    'rite_unet':  ['chase', 'hrf'],
}

EXTRA_KEYS = {
    'lr': 0.0001,          # rieng cho thich nghi, khac lr Stage 0 (0.001)
    'stage1': 1,
    'stage2': 5,
    'teachers': [
        {'name': 'fast', 'keep_rate': 0.99},
        {'name': 'slow', 'keep_rate': 0.999},
    ],
}


def main():
    for source in SOURCES:
        base_path = os.path.join('models', source, 'config.yml')
        if not os.path.exists(base_path):
            print(f"[BO QUA] Khong tim thay {base_path}")
            continue
        with open(base_path, 'r') as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)

        for target in TARGETS.get(source, []):
            new_config = copy.deepcopy(base_config)
            new_config.update(copy.deepcopy(EXTRA_KEYS))

            out_path = os.path.join('models', source, f'config_{target}_dualema.yml')
            with open(out_path, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
            print(f"[OK] Da tao {out_path}")


if __name__ == '__main__':
    main()
