"""
frangi_prior.py
================
Frangi vesselness filter - nguon thong tin CO DIEN, KHONG can hoc, KHONG
phu thuoc model nguon hay domain shift - dua thuan tuy tren hinh hoc cua
cau truc dang ong (phan tich tri rieng ma tran Hessian da ti le).

Dung lam "trong tai thu 3" trong selective_merge_pseudo_label: o vung model
nguon (Teacher_R) nghi ngo (entropy cao), neu Frangi cho tin hieu mach mau
manh, uu tien tin Frangi de va false-negative - dac biet huu ich cho mao
mach manh, tuong phan thap (diem yeu nhat cua model hoc tu du lieu bi lech
domain).

Phu thuoc: scikit-image (pip install scikit-image --break-system-packages)
"""
import numpy as np
import torch
from skimage.filters import frangi
from skimage.exposure import rescale_intensity


def compute_frangi_map(image_gray: np.ndarray, scale_range=(1, 6), scale_step=1,
                        black_ridges=True) -> np.ndarray:
    """Tinh Frangi vesselness tren 1 anh grayscale (numpy, gia tri [0,1] hoac [0,255]).

    Args:
        image_gray: (H, W) numpy array, anh xam (thuong dung kenh green cua
            fundus vi mach mau tuong phan ro nhat o kenh nay).
        scale_range: khoang sigma cho phan tich da ti le - can dieu chinh
            theo do day mach mau thuc te trong dataset (KHONG dung mu quang
            gia tri mac dinh, giong bai hoc voi cldice_num_iter truoc day).
        black_ridges: True neu mach mau TOI hon nen (dung cho fundus RGB
            thong thuong sau khi lay kenh green va dao nguoc neu can - kiem
            tra truc quan truoc khi tin ket qua).

    Returns:
        vesselness map (H, W), gia tri [0, 1], cang cao cang chac la mach mau.
    """
    img = image_gray.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    sigmas = range(scale_range[0], scale_range[1] + 1, scale_step)
    vesselness = frangi(img, sigmas=sigmas, black_ridges=black_ridges)
    vesselness = rescale_intensity(vesselness, out_range=(0, 1))
    return vesselness


def frangi_map_to_tensor(vesselness: np.ndarray, device='cuda') -> torch.Tensor:
    """Chuyen numpy map (H,W) thanh tensor (1,1,H,W) de dung chung voi pipeline torch."""
    t = torch.from_numpy(vesselness).float().unsqueeze(0).unsqueeze(0)
    return t.to(device)


def frangi_assisted_merge(pred_region: torch.Tensor, pred_topology: torch.Tensor,
                           entropy_region: torch.Tensor, frangi_map: torch.Tensor,
                           lambda_thresh=(0.3, 0.5), frangi_confidence_thresh=0.5,
                           num_iter=10):
    """Mo rong selective_merge_pseudo_label: them Frangi lam "trong tai thu 3"
    o vung Teacher_R (region) nghi ngo, khong chi dua vao Teacher_T (topology).

    Logic: lay pred_region lam nen. O vung nghi ngo (entropy cao, xac suat
    mo ho) - NEU Teacher_T HOAC Frangi cho tin hieu manh, thi va vao. Frangi
    dac biet huu ich o mao mach mo, tuong phan thap ma CA HAI teacher (deu
    hoc tu model nguon bi lech domain) co the cung bo sot.
    """
    from region_topology_teacher import soft_skeletonize

    lam1, lam2 = lambda_thresh
    uncertain_mask = ((pred_region > lam1) & (pred_region < lam2)).float()

    skel_topology = soft_skeletonize(pred_topology, num_iter)
    patch_from_topology = (skel_topology > 0.5).float() * uncertain_mask

    frangi_confident = (frangi_map > frangi_confidence_thresh).float()
    patch_from_frangi = frangi_confident * uncertain_mask

    # Hop nhat 2 nguon va (topology + frangi), uu tien nguon nao manh hon
    # tai dung pixel do (lay max thay vi cong don, tranh khuech dai gia).
    patch_signal = torch.maximum(patch_from_topology, patch_from_frangi)
    patch_value = torch.maximum(pred_topology * patch_from_topology,
                                 frangi_map * patch_from_frangi)

    merged = torch.clamp(pred_region + patch_signal * patch_value, 0.0, 1.0)
    return merged
