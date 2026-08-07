"""
unsupervised_early_stop.py
=============================
Early-stopping KHONG dung label dich - chi dua tren thong ke DU DOAN cua
chinh model (khong can ground truth). Y tuong: neu model dang suy thoai
(nhu quan sat duoc khi keo dai Stage II qua 20 epoch - Dice sup tu 0.581
xuong 0.508), thuong di kem voi TY LE PIXEL DU DOAN DUONG TINH thay doi
DOT NGOT (over-segment hoac collapse ve toan nen) - day la tin hieu co
the do ma KHONG can label, dung dung tinh than cua diagnose_gap_v2.py.

KHOA CHAT viec khong dung label: thay vi tai su dung train_loader (von
doc CA anh LAN mask tu dataset.py, chi lo di mask qua dau '_') - lop
ImageOnlyDataset ben duoi CHI DOC FILE ANH, KHONG BAO GIO MO thu muc
mask - ve mat cau truc KHONG THE nao lo label vao duoc, du code sau nay
co sua sai the nao di nua.

Thuat toan:
  1. Sau moi epoch, do ty le pixel du doan duong tinh tren tap anh dich
     (qua ImageOnlyDataset - KHONG CO KHAI NIEM mask trong luong du lieu nay).
  2. So sanh voi trung binh cac epoch TRUOC DO (khong tinh epoch hien tai).
  3. Neu do lech tuong doi > nguong -> danh dau 1 epoch "bat on".
  4. Neu so epoch bat on LIEN TIEP dat patience -> DUNG, quay ve checkpoint
     cua epoch ON DINH GAN NHAT (luu san moi khi phat hien on dinh).
"""
import os
import copy
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import Dataset as _OriginalDataset


class ImageOnlyDataset(_OriginalDataset):
    """
    KE THUA truc tiep tu dataset.Dataset goc - thay vi viet lai transform
    pipeline tu dau (cach lam CU cua file nay, da gay 2 loi thuc te: thieu
    Normalize, thieu RandomRotate90/Flip, khien anh dua vao model lech
    phan bo so voi luc train). Ke thua dam bao PIXEL-FOR-PIXEL giong het
    Dataset goc vi DUNG CHUNG dung 1 doan code transform, khong co nguy co
    lech do viet lai thu cong.

    CHI ghi de __getitem__ de BO MASK khoi gia tri tra ve - dam bao class
    nay khong the "lo" mask ra cho code ben ngoai (vd early-stop), du ben
    trong van phai doc file mask (vi ke thua init tu Dataset goc, can
    mask_dir hop le) - day la danh doi hop ly: uu tien TRUNG THUC ve tien
    xu ly hon la "sach tuyet doi" ve viec khong dung toi file mask.
    """

    def __getitem__(self, idx):
        img, mask, img_id = super().__getitem__(idx)
        return img  # CHI tra ve anh, khong bao gio tra ve mask


class UnsupervisedEarlyStopper:
    def __init__(self, patience=3, ratio_change_threshold=0.20):
        self.patience = patience
        self.ratio_change_threshold = ratio_change_threshold
        self.ratio_history = []
        self.best_state_dict = None
        self.best_epoch = -1
        self.unstable_streak = 0
        self._loader = None  # khoi tao 1 lan duy nhat trong check(), tai su dung

    @torch.no_grad()
    def _predicted_positive_ratio(self, model, image_only_loader):
        model.eval()
        total_ratio, n = 0.0, 0
        for input in image_only_loader:   # CHI co anh - KHONG CO CHO cho mask/label
            input = input.cuda()
            output = model(input)
            prob = torch.sigmoid(output)
            ratio = (prob > 0.5).float().mean().item()
            total_ratio += ratio * input.size(0)
            n += input.size(0)
        model.train()
        return total_ratio / n

    def check(self, model, image_only_loader, epoch_idx, verbose=True):
        """Goi sau MOI epoch. `image_only_loader` PHAI la DataLoader boc
        ImageOnlyDataset (khong phai train_loader thuong). Tra ve True neu
        nen DUNG training tai day."""
        ratio = self._predicted_positive_ratio(model, image_only_loader)
        self.ratio_history.append(ratio)

        if len(self.ratio_history) == 1:
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch_idx
            if verbose:
                print(f"  [EarlyStop-KGS] epoch {epoch_idx}: ty le du doan={ratio:.4f} (epoch dau, luu lam moc)")
            return False

        baseline = sum(self.ratio_history[:-1]) / len(self.ratio_history[:-1])
        relative_change = abs(ratio - baseline) / (baseline + 1e-6)
        is_stable = relative_change <= self.ratio_change_threshold

        if is_stable:
            self.unstable_streak = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch_idx
        else:
            self.unstable_streak += 1

        if verbose:
            status = 'ON DINH' if is_stable else f'BAT ON ({self.unstable_streak}/{self.patience})'
            print(f"  [EarlyStop-KGS] epoch {epoch_idx}: ty le du doan={ratio:.4f}, "
                  f"trung binh truoc do={baseline:.4f}, lech={relative_change*100:.1f}% -> {status}")

        return self.unstable_streak >= self.patience

    def restore_best(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        return self.best_epoch
