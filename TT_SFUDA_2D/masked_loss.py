"""
masked_loss.py
================
BCE+Dice loss CO MASK - dung cho ensemble_mode='trust_region' trong
multi_teacher.py. Khac voi losses.BCEDiceLoss goc (khong ho tro mask),
ham nay CHI tinh loss tren vung trust_mask=1 (noi 2 teacher dong y),
BO QUA hoan toan vung trust_mask=0 (noi bat dong) - khong ep student hoc
theo huong nao ca o vung mo ho.

Cong thuc giu dung tinh than BCEDiceLoss goc (0.5*BCE + Dice), chi them
mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, output_logits, target, mask):
        """
        output_logits: du doan cua student (truoc sigmoid), shape [N,1,H,W]
        target: pseudo-label nhi phan (0/1), cung shape
        mask: 1=tin tuong (tinh loss), 0=bo qua, cung shape
        """
        # --- BCE co mask, chi trung binh tren vung mask=1 ---
        bce = F.binary_cross_entropy_with_logits(output_logits, target, reduction='none')
        bce_masked = (bce * mask).sum() / (mask.sum() + self.smooth)

        # --- Dice co mask ---
        prob = torch.sigmoid(output_logits)
        prob_m = prob * mask
        target_m = target * mask
        intersection = (prob_m * target_m).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (prob_m.sum() + target_m.sum() + self.smooth)

        return 0.5 * bce_masked + dice_loss
