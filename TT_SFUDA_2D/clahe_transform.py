"""
clahe_transform.py
=====================
CLAHE dung CHUAN cho anh mau (khac voi ap CLAHE truc tiep len tung kenh
BGR rieng le - se lam bien dang mau): chuyen sang khong gian mau LAB, CHI
ap dung CLAHE len kenh L (do sang), GIU NGUYEN kenh a,b (thong tin mau).

Viet duoi dang albumentations.ImageOnlyTransform de CAM THANG vao Compose([...])
DA CO SAN trong tt_sfuda_2d_dualema.py / tt_sfuda_2d_region_topology.py, KHONG
can sua dataset.py (dung chung cho nhieu script, sua truc tiep rui ro cao hon).

CO SO: da validate ZERO-SHOT truoc do (validate_clahe_prior.py) - CLAHE giup
RAT LON cho domain shift co domain gap lon (HRF->RITE: +0.19 Dice, 5/5 anh
deu duong), NHUNG gan nhu khong doi/nhe am cho domain gap da nho (HRF->CHASE:
-0.002) - vi vay KHONG bat mac dinh cho MOI domain shift, chi bat co chon
loc qua flag --use_clahe, tuong tu cach resolution duoc chinh RIENG cho
tung domain shift.

Cach dung (them vao Compose da co, khong thay the gi ca):
    from clahe_transform import ClaheLAB
    train_transform = Compose([
        RandomRotate90(), transforms.Flip(),
        ClaheLAB(clip_limit=2.0),      # << THEM dong nay, dat TRUOC Resize/Normalize
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
"""
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class ClaheLAB(ImageOnlyTransform):
    """CLAHE tren kenh L (LAB) - giu nguyen mau, chi chuan hoa do tuong
    phan cuc bo. always_apply=True vi day la buoc CHUAN HOA (giong
    transforms.Normalize()), khong phai augmentation ngau nhien - can ap
    dung NHAT QUAN moi lan (train LAN val), khong duoc random skip."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8), p: float = 1.0):
        # KHONG dung always_apply - phien ban albumentations moi tren Kaggle
        # cua ban khong con nhan tham so nay (da xac nhan qua test thuc te),
        # p=1.0 la du de dam bao LUON ap dung (giong always_apply=True).
        super().__init__(p=p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # dataset.py doc anh o BGR (khong chuyen RGB) - cv2.COLOR_BGR2LAB
        # la dung, GIU NGUYEN quy uoc kenh mau hien co cua pipeline.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def get_transform_init_args_names(self):
        return ('clip_limit', 'tile_grid_size')
