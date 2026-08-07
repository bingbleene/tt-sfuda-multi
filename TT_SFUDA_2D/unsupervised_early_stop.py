"""
unsupervised_early_stop.py (v2)
==================================
SUA LOI THUC TE phat hien duoc: ban v1 dung (a) trung binh TOAN BO lich su
lam baseline - cang nhieu epoch cang bi lam muot/pha loang, mat do nhay voi
xu huong dai han; (b) yeu cau 3 epoch BAT ON LIEN TIEP - tin hieu nhieu
(dao dong manh moi epoch) khien chuoi lien tiep khong bao gio du dai, du
co xu huong troi dat that su qua nhieu epoch (xac nhan bang du lieu that:
chuoi ty le tang dan tu ~0.13 (10 epoch dau) len ~0.17 (10 epoch sau) -
troi dat ro rang, nhung KHONG epoch nao bi bat vi luon bi ngat giua chung
boi 1 epoch "on dinh" ngau nhien).

Sua bang 2 thay doi:
  1. Baseline = trung binh cua N epoch DAU TIEN (co dinh, khong tiep tuc
     cap nhat theo thoi gian) - phat hien DUNG xu huong "da di xa khoi
     diem khoi dau on dinh" thay vi so voi trung binh dang tu lam muot.
  2. Dung "K bat on trong M epoch gan nhat" (khong bat buoc LIEN TIEP) thay
     vi "N lien tiep" - chiu duoc nhieu ngau nhien tung epoch, van bat duoc
     xu huong dai han.

Y tuong tin hieu (ty le pixel du doan duong tinh, khong can label) giu
nguyen - chi sua CACH DIEN GIAI tin hieu do.
"""
import os
import copy
from collections import deque
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Dataset as _OriginalDataset


class ImageOnlyDataset(_OriginalDataset):
    """
    KE THUA truc tiep tu dataset.Dataset goc - dam bao PIXEL-FOR-PIXEL giong
    het (dung chung code transform, khong co nguy co lech do viet lai thu
    cong). CHI ghi de __getitem__ de BO MASK khoi gia tri tra ve.
    """

    def __getitem__(self, idx):
        img, mask, img_id = super().__getitem__(idx)
        return img  # CHI tra ve anh, khong bao gio tra ve mask


class UnsupervisedEarlyStopper:
    def __init__(self, warmup_epochs=3, window_size=5, unstable_count_threshold=3,
                 ratio_change_threshold=0.20):
        """
        warmup_epochs: so epoch DAU dung de tinh baseline CO DINH (khong
            doi sau do) - can it nhat warmup_epochs epoch truoc khi bat dau
            danh gia on dinh/bat on.
        window_size: kich thuoc cua so truot de dem so epoch bat on GAN DAY
            (khong yeu cau lien tiep).
        unstable_count_threshold: so epoch bat on TRONG CUA SO GAN NHAT de
            kich hoat dung (vd 3 trong 5 epoch gan nhat, khong can lien tiep).
        """
        self.warmup_epochs = warmup_epochs
        self.window_size = window_size
        self.unstable_count_threshold = unstable_count_threshold
        self.ratio_change_threshold = ratio_change_threshold

        self.ratio_history = []
        self.baseline = None   # CO DINH sau khi du warmup_epochs, KHONG doi nua
        self.recent_stability = deque(maxlen=window_size)  # True=on dinh, False=bat on

        self.best_state_dict = None
        self.best_epoch = -1

    @torch.no_grad()
    def _predicted_positive_ratio(self, model, image_only_loader):
        model.eval()
        total_ratio, n = 0.0, 0
        for input in image_only_loader:
            input = input.cuda()
            output = model(input)
            prob = torch.sigmoid(output)
            ratio = (prob > 0.5).float().mean().item()
            total_ratio += ratio * input.size(0)
            n += input.size(0)
        model.train()
        return total_ratio / n

    def check(self, model, image_only_loader, epoch_idx, verbose=True):
        """Goi sau MOI epoch. Tra ve True neu nen DUNG training tai day."""
        ratio = self._predicted_positive_ratio(model, image_only_loader)
        self.ratio_history.append(ratio)

        # ==== Giai doan warmup: chua du du lieu de co baseline on dinh ====
        if len(self.ratio_history) <= self.warmup_epochs:
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch_idx
            if len(self.ratio_history) == self.warmup_epochs:
                # dung TRUNG VI (median) thay vi trung binh - do bot nhay
                # voi 1 epoch bat thuong don le trong giai doan warmup ngan
                sorted_hist = sorted(self.ratio_history)
                mid = len(sorted_hist) // 2
                self.baseline = (sorted_hist[mid] if len(sorted_hist) % 2 == 1
                                 else (sorted_hist[mid-1] + sorted_hist[mid]) / 2)
            if verbose:
                print(f"  [EarlyStop-KGS] epoch {epoch_idx}: ty le du doan={ratio:.4f} "
                      f"(warmup {len(self.ratio_history)}/{self.warmup_epochs})")
            return False

        # ==== Sau warmup: so voi baseline CO DINH (khong doi theo thoi gian) ====
        relative_change = abs(ratio - self.baseline) / (self.baseline + 1e-6)
        is_stable = relative_change <= self.ratio_change_threshold
        self.recent_stability.append(is_stable)

        if is_stable:
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch_idx

        unstable_in_window = sum(1 for s in self.recent_stability if not s)

        if verbose:
            status = 'ON DINH' if is_stable else 'BAT ON'
            print(f"  [EarlyStop-KGS] epoch {epoch_idx}: ty le du doan={ratio:.4f}, "
                  f"baseline CO DINH={self.baseline:.4f}, lech={relative_change*100:.1f}% -> {status} "
                  f"(bat on {unstable_in_window}/{len(self.recent_stability)} trong {self.window_size} epoch gan nhat)")

        return (len(self.recent_stability) == self.window_size and
                unstable_in_window >= self.unstable_count_threshold)

    def restore_best(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        return self.best_epoch
