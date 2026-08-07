"""
Region-Teacher / Topology-Teacher extension cho TT-SFUDA Stage II.

Gồm 2 phần:
1. SoftClDiceLoss — loss centerline-Dice (Shit et al., CVPR 2021), dùng cho
   Pair T (Topology Teacher) để phạt lỗi đứt đoạn mạch máu mà Dice/BCE thường bỏ qua.
2. selective_merge_pseudo_label — kết hợp prediction của Teacher_R và Teacher_T
   theo logic Selective Voting (bám sát Eqn 5-7 của paper TT-SFUDA gốc), thay vì
   trộn toàn ảnh theo trọng số cố định.

Hai phần này ĐỘC LẬP với pipeline hiện có — không sửa multi_teacher.py gốc,
để test riêng trước khi tích hợp (đúng workflow bạn đang theo: file mới đặt
flat trong TT_SFUDA_2D/, không đụng code cũ).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Soft clDice loss
# ---------------------------------------------------------------------------

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Xấp xỉ phép co hình thái học (erosion) bằng min-pooling, khả vi.
    img: (B, 1, H, W), giá trị trong [0, 1] (đã qua sigmoid).
    """
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Xấp xỉ phép giãn hình thái học (dilation) bằng max-pooling."""
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    """Xấp xỉ khả vi của skeleton/centerline (thuật toán trong paper clDice gốc).

    Lặp num_iter lần erosion + so sánh phần dư sau khi mở (opening) để dần
    "bào mỏng" vùng dự đoán về đường tâm 1-pixel. num_iter nên đủ lớn để phủ
    độ dày lớn nhất của mạch máu trong ảnh (mạch máu chính ở fundus thường
    không quá 8-10px nên 10 iter là hợp lý, có thể chỉnh theo dataset).
    """
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(num_iter):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """Soft clDice loss cho cấu trúc dạng ống (tubular structures).

    clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens)
    trong đó:
      Tprec = |skeleton(pred) ∩ mask_gt| / |skeleton(pred)|   (độ chính xác)
      Tsens = |skeleton(gt) ∩ mask_pred| / |skeleton(gt)|     (độ nhạy)

    Dùng pseudo-label làm "gt" trong bối cảnh SFUDA (không có nhãn thật).
    Kết hợp với Dice/BCE thông thường, KHÔNG thay thế hoàn toàn, vì clDice
    một mình không phạt sai lệch vùng/diện tích.
    """

    def __init__(self, num_iter: int = 10, smooth: float = 1e-6):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred_logits: (B, 1, H, W) — logits chưa qua sigmoid (student output).
        target: (B, 1, H, W) — pseudo-label, giá trị trong [0, 1] (có thể soft).
        """
        pred = torch.sigmoid(pred_logits)

        skel_pred = soft_skeletonize(pred, self.num_iter)
        skel_true = soft_skeletonize(target, self.num_iter)

        tprec = (torch.sum(skel_pred * target) + self.smooth) / (
            torch.sum(skel_pred) + self.smooth
        )
        tsens = (torch.sum(skel_true * pred) + self.smooth) / (
            torch.sum(skel_true) + self.smooth
        )

        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice


class TopologyTeacherLoss(nn.Module):
    """Loss tổng hợp cho Pair T (Topology Teacher): Dice + BCE + lambda * clDice.

    Giữ Dice+BCE làm nền để không mất khả năng bắt đúng vùng (clDice một mình
    không đủ, dễ overfit vào các đường mảnh nhiễu). lambda_cl cần sweep —
    KHÔNG giả định trước giá trị, theo đúng bài học fg_weight=6.0 trước đây
    (phải dò, không đoán).
    """

    def __init__(self, lambda_cl: float = 1.0, num_iter: int = 10):
        super().__init__()
        self.cldice = SoftClDiceLoss(num_iter=num_iter)
        self.lambda_cl = lambda_cl

    def forward(self, pred_logits, target, base_seg_loss_fn):
        """
        base_seg_loss_fn: hàm Lseg gốc đã có sẵn trong codebase (BCE + Dice),
        truyền vào để tái dùng, không viết lại (đúng bài học "kế thừa thay vì
        viết lại" đã rút ra ở phần early-stopping).
        """
        seg_loss = base_seg_loss_fn(pred_logits, target)
        cl_loss = self.cldice(pred_logits, target)
        return seg_loss + self.lambda_cl * cl_loss


# ---------------------------------------------------------------------------
# 2. Selective merge pseudo-label (Region-Teacher + Topology-Teacher)
# ---------------------------------------------------------------------------

def selective_merge_pseudo_label(
    pred_region: torch.Tensor,
    pred_topology: torch.Tensor,
    entropy_region: torch.Tensor,
    lambda_thresh: tuple = (0.3, 0.5),
    num_iter: int = 10,
) -> torch.Tensor:
    """Kết hợp prediction của Teacher_R và Teacher_T theo logic Selective
    Voting (bám Eqn 5-7 paper gốc), thay vì trộn toàn ảnh theo trọng số cố định.

    Ý tưởng: lấy pred_region làm nền. Ở vùng "nghi đứt đoạn" (xác định qua
    entropy cao VÀ nằm trên/gần skeleton), nếu Teacher_T tự tin có mạch máu
    tại đó, thì OR thêm vào để vá chỗ hổng.

    Args:
        pred_region: (B,1,H,W) sigmoid probability từ Teacher_R (weak-aug input).
        pred_topology: (B,1,H,W) sigmoid probability từ Teacher_T (weak-aug input).
        entropy_region: (B,1,H,W) entropy map của pred_region (dùng lại công
            thức Eqn 2 của paper gốc: -p*log(p) - (1-p)*log(1-p)).
        lambda_thresh: (lambda1, lambda2) ngưỡng vùng "false negative nghi ngờ",
            mặc định lấy theo giá trị paper gốc (0.3, 0.5) làm điểm khởi đầu —
            CẦN sweep lại cho bối cảnh 2-teacher này, không dùng mù quáng.
        num_iter: số vòng lặp soft-skeletonize.

    Returns:
        merged: (B,1,H,W) pseudo-label đã vá, giá trị trong [0,1] (dùng làm
            target cho cả hai student, giống cách paper gốc dùng ¯y_t^n).
    """
    lam1, lam2 = lambda_thresh

    # Vùng nghi ngờ đứt đoạn: xác suất region nằm trong khoảng mơ hồ (không
    # chắc là mạch máu, không chắc là nền) — giống Eqn 6 paper gốc.
    uncertain_mask = ((pred_region > lam1) & (pred_region < lam2)).float()

    # Giới hạn vùng vá chỉ quanh skeleton của topology-teacher, tránh vá lan
    # sang vùng nền rộng (khác với trust_region cũ vốn AND toàn vùng và lỡ
    # cắt mất mạch máu mảnh — ở đây CHỈ dùng skeleton để mở rộng, không thu hẹp).
    skel_topology = soft_skeletonize(pred_topology, num_iter)
    patch_signal = (skel_topology > 0.5).float() * uncertain_mask

    # OR: giữ nguyên toàn bộ pred_region, chỉ thêm vào chỗ patch_signal bật.
    merged = torch.clamp(pred_region + patch_signal * pred_topology, 0.0, 1.0)
    return merged
