"""
_compat.py
==========
Va tuong thich cho albumentations phien ban moi (>=1.4, vd 2.0.8 tren Kaggle
hien tai) - phien ban nay da XOA HOAN TOAN submodule
`albumentations.augmentations.transforms` va class `Flip` (random flip
ngang/doc), trong khi code goc cua tac gia (tt_sfuda_2d.py) viet cho ban cu
(~2021) van dung cu phap:

    from albumentations.augmentations import transforms
    transforms.Flip()
    transforms.Normalize()

File nay KHONG sua tt_sfuda_2d.py - chi "vien" lai submodule da mat bang
cach chen 1 module gia vao sys.modules TRUOC khi tt_sfuda_2d.py chay dong
import cua no. Phai duoc import (hoac chay qua run.py) TRUOC bat ky import
albumentations nao khac trong tien trinh.

Flip() duoc mo phong lai bang OneOf(HorizontalFlip, VerticalFlip) - tuong
duong ve mat chuc nang (ngau nhien lat ngang HOAC doc), khong hoan toan
giong RNG goc cua albumentations cu nhung khong anh huong ban chat augmentation.
Normalize() van con nguyen ham albumentations top-level, khong doi hanh vi.
"""
import sys
import types

import albumentations as A

_fake = types.ModuleType('albumentations.augmentations.transforms')
_fake.Normalize = A.Normalize
_fake.Flip = lambda always_apply=False, p=0.5: A.OneOf(
    [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=p
)
sys.modules['albumentations.augmentations.transforms'] = _fake


# ============================================================
# Va 2: torchvision RandomSolarize - ban torchvision moi (>=0.17) kiem tra
# nghiem ngat threshold < bound cua anh (bound=1.0 cho anh float, 255.0 cho
# uint8). Code goc dung threshold=192.0 (quy uoc uint8 [0,255] cua ban cu),
# nhung anh thuc te trong pipeline la tensor float [~0,1] (sau Normalize
# cua albumentations + /255 trong dataset.py) -> vuot bound, bi TypeError.
#
# Vien: khi threshold >= bound cua anh float, QUY DOI TY LE tuong duong
# (threshold/255 * bound) thay vi bo qua hoan toan buoc solarize - giu dung
# tinh than "solarize 1 phan anh" cua augmentation goc thay vi vo hieu hoa no.
# ============================================================
import torch
import torchvision.transforms.functional as _TF

_original_solarize = _TF.solarize


def _patched_solarize(img, threshold):
    if isinstance(img, torch.Tensor) and img.is_floating_point():
        bound = 1.0
        if threshold >= bound:
            threshold = min((threshold / 255.0) * bound, bound - 1e-6)
    return _original_solarize(img, threshold)


_TF.solarize = _patched_solarize