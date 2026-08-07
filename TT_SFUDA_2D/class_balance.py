"""
class_balance.py
==================
Cai bien tu ky thuat "Global Knowledge Guided Loss Calibration" trong CBMT
(Tang et al., Source-Free Domain Adaptive Fundus Image Segmentation with
Class-Balanced Mean Teacher, MICCAI 2023, arXiv:2307.09973 - Eq. 5-6).

Van de goc (CBMT Section 2.2, va DUNG voi vessel segmentation cua chung ta):
tien canh (mach mau) chi chiem ~5-8% pixel (do bang diagnose_gap_v2.py).
Neu tinh BCE binh thuong, so hang nen (background) se AP DAO loss, lam
loang tin hieu hoc cho tien canh.

Cong thuc goc (Eq. 5): tinh loss BCE trung binh RIENG cho pixel tien canh
va hau canh, dua tren PSEUDO-LABEL (khong phai ground truth - vi day la
SFUDA, khong co label that o dich):
    eta_fg = trung binh loss tren pixel co pseudo_label=1
    eta_bg = trung binh loss tren pixel co pseudo_label=0

Eq. 6: dung ty le eta_fg/eta_bg de CAN CHINH trong so so hang nen:
    L_calibrated = E[ y*log(p) + (eta_fg/eta_bg)*(1-y)*log(1-p) ]

Khac biet nho so voi ban CBMT goc: thay vi tinh lai eta tren TOAN BO
dataset moi epoch (can 2-pass), ta dung EMA (exponential moving average)
CAP NHAT LIEN TUC qua tung iteration - phu hop hon voi tap du lieu nho
(20-35 anh) va so epoch it (5-10) cua TT-SFUDA, tranh phai cho het 1 epoch
moi co so lieu dau tien de hieu chinh.

Bo loc pixel "khong nhieu thong tin" (CBMT dung nguong alpha phuc tap dua
tren |p-gamma|/|y-gamma|) duoc DON GIAN HOA thanh: chi tinh eta tren pixel
co xac suat KHONG qua cuc tri (0.05 < p < 0.95) - dat duoc cung muc dich
(loai pixel da qua "chac chan", khong dong gop thong tin) ma khong can
tai hien chinh xac cong thuc loc phuc tap cua ban goc. CBMT tu bao cao ket
qua ON DINH voi nhieu gia tri nguong loc khac nhau (Table 3 trong paper),
nen don gian hoa nay duoc coi la chap nhan duoc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalanceTracker:
    def __init__(self, momentum=0.9, prob_low=0.05, prob_high=0.95):
        self.momentum = momentum
        self.prob_low = prob_low
        self.prob_high = prob_high
        self.eta_fg = None
        self.eta_bg = None

    @torch.no_grad()
    def update(self, output_logits, pseudo_label):
        """Goi moi iteration VOI DU LIEU CUA CHINH ITERATION DO (khong can
        cho het epoch) - cap nhat eta_fg/eta_bg qua EMA."""
        prob = torch.sigmoid(output_logits)
        per_pixel_bce = F.binary_cross_entropy_with_logits(output_logits, pseudo_label, reduction='none')

        informative = ((prob > self.prob_low) & (prob < self.prob_high)).float()
        fg_mask = pseudo_label * informative
        bg_mask = (1 - pseudo_label) * informative

        fg_cnt = fg_mask.sum().item()
        bg_cnt = bg_mask.sum().item()
        if fg_cnt < 1 or bg_cnt < 1:
            return  # khong du pixel "thong tin" trong batch nay, giu nguyen eta cu

        batch_eta_fg = (per_pixel_bce * fg_mask).sum().item() / fg_cnt
        batch_eta_bg = (per_pixel_bce * bg_mask).sum().item() / bg_cnt

        if self.eta_fg is None:
            self.eta_fg, self.eta_bg = batch_eta_fg, batch_eta_bg
        else:
            self.eta_fg = self.momentum * self.eta_fg + (1 - self.momentum) * batch_eta_fg
            self.eta_bg = self.momentum * self.eta_bg + (1 - self.momentum) * batch_eta_bg

    def get_ratio(self, eps=1e-6):
        """Ty le eta_fg/eta_bg dung de can chinh trong so nen (Eq.6).
        Tra ve 1.0 (khong can chinh gi) neu chua co du lieu."""
        if self.eta_fg is None or self.eta_bg is None:
            return 1.0
        return self.eta_fg / (self.eta_bg + eps)


class CalibratedBCEDiceLoss(nn.Module):
    """Ban thay the BCEDiceLoss goc (losses.py), co THEM buoc can chinh
    trong so nen theo Eq.6 CBMT. Dice term giu nguyen KHONG can chinh (Dice
    von da it nhay voi mat can bang so luong pixel hon BCE)."""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, output_logits, target, bg_weight_ratio=1.0):
        prob = torch.sigmoid(output_logits)
        eps = 1e-8
        prob_c = prob.clamp(eps, 1 - eps)

        fg_term = target * torch.log(prob_c)
        bg_term = bg_weight_ratio * (1 - target) * torch.log(1 - prob_c)
        bce_calibrated = -(fg_term + bg_term).mean()

        intersection = (prob * target).sum()
        dice = 1 - (2 * intersection + self.smooth) / (prob.sum() + target.sum() + self.smooth)

        return 0.5 * bce_calibrated + dice
