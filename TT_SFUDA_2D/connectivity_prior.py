"""
connectivity_prior.py
========================
Y TUONG KHAC BAN CHAT so voi 3 huong da thu (cross-arch/wavelet/SAM) - do
KHONG phai "them 1 bo du doan doc lap roi vote", ma la HAU XU LY tat dinh
dua tren 1 SU THAT GIAI PHAU: mach mau vong mac la MOT CAY LIEN THONG DUY
NHAT, goc o dia thi (khong co "hon dao" mach mau roi rac ve mat sinh hoc
that). clDice (da dung) chi phat dut doan CUC BO qua skeleton, KHONG ep
toan cuc "phai la 1 cay" - conected-component analysis lam dung dieu do,
va KHONG can hoc gi, KHONG co gradient nen KHONG the xung dot voi bat ky
loss nao khac (khac han moi thu nghiem truoc).

Uu diem so voi 3 huong da dong:
  - Khong can train lai gi - ap dung TRUC TIEP len output cua model DA CO
    (validate duoc trong vai phut, khong can GPU dai).
  - Khong co "chat luong nguon thong tin" de lo (khong phai predictor moi
    co the sai) - chi loai bo pixel model DA TU CHO LA duong tinh nhung
    KHONG lien thong voi thanh phan chinh, tuc CHi lam Dice TOT HON HOAC
    BANG (khong bao gio lam mat mach mau that dang lien thong voi cay).

RUI RO CAN KIEM TRA (dung suy doan mo): neu mot so mach mau THAT trong GT
cung bi dut roi (do domain shift lam model du doan dut doan nghiem trong),
loai component nho co the VO TINH cat bo ca phan dung. Day chinh la ly do
BAT BUOC validate doc lap truoc (script validate_connectivity_postprocess.py)
thay vi tin ngay vi "nghe co ly".
"""
import numpy as np
from scipy import ndimage


def largest_component_prior(pred_binary: np.ndarray, keep_top_k: int = 1,
                              min_component_size: int = 20,
                              dilate_before_merge: int = 3) -> np.ndarray:
    """
    pred_binary: (H,W) mask nhi phan (0/1) tu prediction cua model.
    keep_top_k: giu k thanh phan lien thong LON NHAT (mac dinh 1 - dung
        gia dinh "1 cay duy nhat"; co the tang len 2-3 neu anh bi che
        khuat/cat mep lam cay chinh bi tach doi boi tien xu ly).
    min_component_size: BO QUA rule nay cho thanh phan qua nho (duoi
        nguong nay coi la nhieu chac chan, khong tinh vao top-k) - tranh
        giu nham 1 dom nhieu lon tinh co vi no "lon nhat trong so cac dom nhieu".
    dilate_before_merge: GIAN NO truoc khi tim thanh phan lien thong (khong
        anh huong mask tra ve cuoi) - QUAN TRONG: mach mau mong 1px de bi
        DUT GIA (khong lien thong that trong pixel-space du lien thong that
        ve mat giai phau) do domain shift/nhieu. Gian no vai pixel truoc
        khi gan nhan connected-component giup NOI LAI cac doan gan nhau,
        tranh loai oan cac nhanh that chi vi dut 1-2 pixel.

    Tra ve: mask (H,W) da loc, chi con cac thanh phan lien thong duoc giu.
    """
    if dilate_before_merge > 0:
        struct = ndimage.generate_binary_structure(2, 2)
        dilated = ndimage.binary_dilation(pred_binary, structure=struct,
                                           iterations=dilate_before_merge)
    else:
        dilated = pred_binary

    labeled, num_features = ndimage.label(dilated, structure=np.ones((3, 3)))
    if num_features == 0:
        return pred_binary  # khong co gi de loc, tra ve nguyen ban

    sizes = ndimage.sum(dilated, labeled, range(1, num_features + 1))
    valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_component_size]
    if not valid_labels:
        return pred_binary

    sorted_labels = sorted(valid_labels, key=lambda l: sizes[l - 1], reverse=True)
    keep_labels = set(sorted_labels[:keep_top_k])

    # Mask thanh phan duoc giu O KHONG GIAN DA GIAN NO, roi AND lai voi
    # pred_binary GOC (chua gian no) - dam bao khong THEM pixel moi nao
    # ngoai nhung gi model da tu du doan, CHI loai bo bot.
    keep_mask_dilated = np.isin(labeled, list(keep_labels))
    return (pred_binary.astype(bool) & keep_mask_dilated).astype(np.uint8)
