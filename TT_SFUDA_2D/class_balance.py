"""
class_balance.py (v2)
=======================
Sau khi xac nhan bang thuc nghiem: eta_fg > eta_bg XUYEN SUOT toan bo Stage
II tren ca 2 domain shift target=RITE (khong dao dong, ket qua 20/20 epoch
deu bg_ratio=1.0000 - xem log thuc te) - cong thuc CBMT goc (chi duoc phep
GIAM trong so nen) hoan toan KHONG CO TAC DUNG voi vessel segmentation, vi
tien canh (mach mau) von di kho hon nen trong SUOT qua trinh train, khong
chi luc dau.

File nay cung cap 3 CHIEN LUOC de thu nghiem co kiem soat, chon qua tham so
`strategy`:

  'cbmt'       : cong thuc CBMT GOC (Tang et al. MICCAI 2023, Eq.5-6), chi
                 duoc phep giam trong so nen (ratio in [0.1, 1.0]) - GIU LAI
                 de doi chieu, da xac nhan KHONG CO TAC DUNG voi du lieu nay.

  'symmetric'  : TONG QUAT HOA - ben nao dang co loss trung binh CAO HON
                 (kho hon) thi duoc TANG trong so, khong co dinh "chi nen
                 moi duoc dieu chinh" nhu ban CBMT goc. Voi du lieu vessel
                 (tien canh luon kho hon), ky vong se LUON tang trong so
                 tien canh - dung huong truc giac nhung CHUA CO BANG CHUNG
                 thuc nghiem la co tot hon khong.

  'fixed_fg'   : DON GIAN NHAT - nhan co dinh trong so tien canh voi 1 he so
                 (vd 2.0), KHONG can theo doi thong ke dong. Dung de kiem tra
                 xem do phuc tap "dong" cua 'symmetric' co thuc su can thiet
                 khong, hay chi can 1 hang so co dinh la du.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalanceTracker:
    def __init__(self, strategy='symmetric', momentum=0.9, prob_low=0.05, prob_high=0.95,
                 ratio_min=0.1, ratio_max=1.0, fixed_fg_weight=2.0):
        assert strategy in ('cbmt', 'symmetric', 'fixed_fg')
        self.strategy = strategy
        self.momentum = momentum
        self.prob_low = prob_low
        self.prob_high = prob_high
        self.ratio_max = ratio_max   # chi dung cho strategy='cbmt'
        self.ratio_min = ratio_min   # chi dung cho strategy='cbmt'/'symmetric'
        self.fixed_fg_weight = fixed_fg_weight  # chi dung cho strategy='fixed_fg'
        self.eta_fg = None
        self.eta_bg = None

    @torch.no_grad()
    def update(self, output_logits, pseudo_label):
        if self.strategy == 'fixed_fg':
            return  # khong can theo doi thong ke gi ca

        prob = torch.sigmoid(output_logits)
        per_pixel_bce = F.binary_cross_entropy_with_logits(output_logits, pseudo_label, reduction='none')
        if not torch.isfinite(per_pixel_bce).all():
            return

        informative = ((prob > self.prob_low) & (prob < self.prob_high)).float()
        fg_mask = pseudo_label * informative
        bg_mask = (1 - pseudo_label) * informative

        fg_cnt = fg_mask.sum().item()
        bg_cnt = bg_mask.sum().item()
        if fg_cnt < 1 or bg_cnt < 1:
            return

        batch_eta_fg = (per_pixel_bce * fg_mask).sum().item() / fg_cnt
        batch_eta_bg = (per_pixel_bce * bg_mask).sum().item() / bg_cnt

        if self.eta_fg is None:
            self.eta_fg, self.eta_bg = batch_eta_fg, batch_eta_bg
        else:
            self.eta_fg = self.momentum * self.eta_fg + (1 - self.momentum) * batch_eta_fg
            self.eta_bg = self.momentum * self.eta_bg + (1 - self.momentum) * batch_eta_bg

    def get_weights(self, eps=1e-6):
        """Tra ve (weight_fg, weight_bg) - CACH DUY NHAT de lay trong so,
        thay the get_ratio() cua ban v1 (chi tra ve 1 so cho nen)."""
        if self.strategy == 'fixed_fg':
            return self.fixed_fg_weight, 1.0

        if self.eta_fg is None or self.eta_bg is None:
            return 1.0, 1.0

        raw_ratio = self.eta_fg / (self.eta_bg + eps)
        if not (raw_ratio == raw_ratio) or raw_ratio in (float('inf'), float('-inf')):
            return 1.0, 1.0

        if self.strategy == 'cbmt':
            # CHI duoc giam trong so nen (nhu ban goc), khong bao gio tang
            bg_w = max(self.ratio_min, min(self.ratio_max, raw_ratio))
            return 1.0, bg_w

        else:  # 'symmetric'
            raw_ratio = max(self.ratio_min, min(1.0 / self.ratio_min, raw_ratio))
            if raw_ratio >= 1.0:
                # tien canh dang kho hon -> tang trong so tien canh
                return raw_ratio, 1.0
            else:
                # nen dang kho hon -> tang trong so nen (hiem gap voi vessel,
                # nhung van xu ly dung neu xay ra)
                return 1.0, 1.0 / raw_ratio


class CalibratedBCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, output_logits, target, fg_weight=1.0, bg_weight=1.0):
        fg_weight = max(0.01, min(10.0, fg_weight))
        bg_weight = max(0.01, min(10.0, bg_weight))

        prob = torch.sigmoid(output_logits)
        eps = 1e-6   # PHAI >= ~1e-6, khong duoc dung 1e-8 (bi lam tron mat trong float32 gan 1.0)
        prob_c = prob.clamp(eps, 1 - eps)

        fg_term = fg_weight * target * torch.log(prob_c)
        bg_term = bg_weight * (1 - target) * torch.log(1 - prob_c)
        bce_calibrated = -(fg_term + bg_term).mean()

        intersection = (prob * target).sum()
        dice = 1 - (2 * intersection + self.smooth) / (prob.sum() + target.sum() + self.smooth)

        loss = 0.5 * bce_calibrated + dice
        if not torch.isfinite(loss):
            fg_term = target * torch.log(prob_c)
            bg_term = (1 - target) * torch.log(1 - prob_c)
            bce_fallback = -(fg_term + bg_term).mean()
            loss = 0.5 * bce_fallback + dice
        return loss
