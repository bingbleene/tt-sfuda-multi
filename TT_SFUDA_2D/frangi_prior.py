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
import torch.nn.functional as F
from skimage.exposure import rescale_intensity

# Uu tien dung cuCIM (RAPIDS/NVIDIA) - ban GPU CHINH THUC cua skimage.filters.frangi
# (API giong het, cung cong thuc, chi khac chay tren CuPy/GPU thay vi NumPy/CPU).
# AN TOAN HON code Hessian tu viet tay (co nguy co sai lech cong thuc), vi day
# la thu vien da kiem chung, khong phai tu che. Neu khong cai duoc (Kaggle
# khong co san cuCIM/CuPy tuong thich CUDA), tu dong lui ve skimage (CPU).
try:
    import cupy as cp
    from cucim.skimage.filters import frangi as frangi_gpu
    from cucim.skimage.exposure import rescale_intensity as rescale_intensity_gpu
    _HAS_CUCIM = True
except ImportError:
    _HAS_CUCIM = False

from skimage.filters import frangi


def compute_frangi_map(image_gray: np.ndarray, scale_range=(2, 8), scale_step=1,
                        black_ridges=True, apply_clahe=True,
                        clahe_clip_limit=2.0, clahe_tile_size=(8, 8),
                        use_gpu='auto') -> np.ndarray:
    """Tinh Frangi vesselness tren 1 anh grayscale (numpy, gia tri [0,1] hoac [0,255]).

    DA KIEM CHUNG tren 5 anh RITE that (21-25_training.png), threshold CO
    DINH (khong do rieng tung anh - dung ca nhau moi khi trien khai thuc te):
      - scale_range=(2,8), CLAHE clip=2.0, threshold nhi phan hoa = 0.005
      - Dice trung binh (n=5) = 0.5352, std = 0.0524
      - So voi TT-SFUDA sau adapt tren CHASE->RITE = 0.5263 (Table 1 paper)
        -> Frangi (KHONG hoc gi) gan bang model sau adapt (CO hoc) - xac
        nhan day la nguon thong tin doc lap co gia tri thuc su, dang tich
        hop, khong phai chi la y tuong tren giay.

    Args:
        image_gray: (H, W) numpy array, anh xam (kenh green cua fundus).
        scale_range: (2,8) la gia tri DA KIEM CHUNG cho RITE - CAN kiem tra
            lai rieng tren CHASE/HRF truoc khi dung chung (do phan giai/do
            day mach mau khac nhau giua 3 dataset).
        black_ridges: True (da xac nhan dung voi kenh green sau CLAHE).
        apply_clahe: BAT BUOC True - Dice sut tu 0.535 xuong ~0.15 neu tat.
        use_gpu: 'auto' (dung cuCIM neu co san, tu dong lui ve CPU neu khong),
            True (bat buoc GPU, loi neu khong co cuCIM), False (bat buoc CPU).
            cuCIM la BAN GPU CHINH THUC cua skimage (RAPIDS/NVIDIA), CUNG
            CONG THUC, khong phai code tu viet tay - ket qua ve mat ly
            thuyet giong het ban CPU da kiem chung, chi nhanh hon.

    Returns:
        vesselness map (H, W), gia tri [0, 1].
    """
    img = image_gray.astype(np.float64)
    if img.max() > 1.0:
        img_uint8 = image_gray.astype(np.uint8) if image_gray.dtype != np.uint8 else image_gray
    else:
        img_uint8 = (image_gray * 255).astype(np.uint8)

    if apply_clahe:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
        img_uint8 = clahe.apply(img_uint8)

    img = img_uint8.astype(np.float64) / 255.0
    sigmas = list(range(scale_range[0], scale_range[1] + 1, scale_step))

    run_on_gpu = (use_gpu is True) or (use_gpu == 'auto' and _HAS_CUCIM)
    if use_gpu is True and not _HAS_CUCIM:
        raise RuntimeError("use_gpu=True nhung khong tim thay cuCIM/CuPy. "
                            "Cai bang: pip install cucim-cu12 cupy-cuda12x "
                            "(doi cu12 -> dung phien ban CUDA that tren may).")

    if run_on_gpu:
        img_gpu = cp.asarray(img)
        vesselness_gpu = frangi_gpu(img_gpu, sigmas=sigmas, black_ridges=black_ridges)
        vesselness_gpu = rescale_intensity_gpu(vesselness_gpu, out_range=(0, 1))
        vesselness = cp.asnumpy(vesselness_gpu)
    else:
        vesselness = frangi(img, sigmas=sigmas, black_ridges=black_ridges)
        vesselness = rescale_intensity(vesselness, out_range=(0, 1))
    return vesselness


# Threshold nhi phan hoa CO DINH, dung khi can mask cung (0/1) tu vesselness map.
# Gia tri 0.005 da kiem chung thuc te tren RITE (xem docstring compute_frangi_map).
# CAN kiem tra lai rieng cho CHASE/HRF truoc khi tin dung chung 1 gia tri.
FRANGI_BINARY_THRESHOLD_DEFAULT = 0.005


def _gaussian_2nd_deriv_kernels(sigma, device):
    """Tao 3 kernel dao ham bac 2 cua Gaussian (Gxx, Gyy, Gxy) tai 1 ti le sigma,
    dung de tinh ma tran Hessian bang tich chap (conv2d) - day la cach lam
    TRUC TIEP tren GPU, thay vi skimage (chi chay CPU)."""
    radius = max(3, int(4 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    x, y = torch.meshgrid(coords, coords, indexing='xy')
    s2 = sigma ** 2
    g = torch.exp(-(x**2 + y**2) / (2 * s2)) / (2 * torch.pi * s2)
    gxx = (x**2 - s2) / (s2**2) * g
    gyy = (y**2 - s2) / (s2**2) * g
    gxy = (x * y) / (s2**2) * g
    # Chuan hoa gamma (scale-normalized derivative) de cac ti le so sanh cong bang.
    norm = sigma ** 2
    k = torch.stack([gxx, gyy, gxy]) * norm
    return k.unsqueeze(1)  # (3, 1, K, K) - dung lam conv weight


def compute_frangi_map_gpu(images: torch.Tensor, scale_range=(2, 8), scale_step=1,
                            black_ridges=True, beta=0.5, c=15.0, device='cuda') -> torch.Tensor:
    """Ban GPU cua Frangi vesselness - xu ly CA LOAT anh cung luc (batch),
    nhanh hon nhieu so voi skimage.filters.frangi (chi chay CPU, tung anh 1).

    Args:
        images: (N, 1, H, W) tensor, gia tri [0,1], DA qua CLAHE tu truoc
            (CLAHE van chay tren CPU vi skimage/OpenCV khong co ban GPU don
            gian - nhung day la buoc RE, khong phai nut that thoi gian).
        scale_range, scale_step: giong compute_frangi_map (CPU).
        black_ridges: True neu mach mau toi hon nen (dao nguoc anh truoc khi
            tinh, giong cach skimage xu ly noi bo).
        beta, c: tham so cong thuc Frangi goc (Frangi et al. 1998) - dung gia
            tri mac dinh pho bien, CHUA sweep rieng cho vessel segmentation.

    Returns:
        (N, 1, H, W) tensor vesselness, gia tri [0, 1].
    """
    images = images.to(device)
    if black_ridges:
        images = 1.0 - images

    sigmas = np.arange(scale_range[0], scale_range[1] + 1, scale_step)
    max_response = torch.zeros_like(images)

    for sigma in sigmas:
        kernels = _gaussian_2nd_deriv_kernels(float(sigma), device)
        pad = kernels.shape[-1] // 2
        hessian = F.conv2d(images, kernels, padding=pad)  # (N, 3, H, W): Ixx, Iyy, Ixy
        ixx, iyy, ixy = hessian[:, 0:1], hessian[:, 1:2], hessian[:, 2:3]

        # Tri rieng cua ma tran Hessian 2x2 [[Ixx, Ixy],[Ixy, Iyy]] - cong thuc
        # dong dai so, tinh truc tiep khong can eigh (nhanh hon nhieu tren GPU).
        tmp = torch.sqrt((ixx - iyy) ** 2 + 4 * ixy ** 2 + 1e-12)
        lambda1 = 0.5 * (ixx + iyy + tmp)
        lambda2 = 0.5 * (ixx + iyy - tmp)
        # Sap xep theo |lambda1| <= |lambda2| (dung quy uoc Frangi goc).
        swap = lambda1.abs() > lambda2.abs()
        l1 = torch.where(swap, lambda2, lambda1)
        l2 = torch.where(swap, lambda1, lambda2)

        rb = l1 / (l2 + 1e-12)
        s = torch.sqrt(l1 ** 2 + l2 ** 2)
        vesselness = torch.exp(-(rb ** 2) / (2 * beta ** 2)) * \
            (1 - torch.exp(-(s ** 2) / (2 * c ** 2)))
        vesselness = torch.where(l2 > 0, torch.zeros_like(vesselness), vesselness)

        max_response = torch.maximum(max_response, vesselness)

    # Chuan hoa ve [0,1] theo tung anh (giong rescale_intensity cua ban CPU).
    flat = max_response.view(max_response.size(0), -1)
    mn = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    mx = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    max_response = (max_response - mn) / (mx - mn + 1e-12)
    return max_response


def precompute_frangi_maps_gpu(img_dir: str, img_ids: list, img_ext: str,
                                input_h: int, input_w: int, cache_path: str = None,
                                batch_size: int = 8, device: str = 'cuda'):
    """Ban GPU cua precompute_frangi_maps - CLAHE van chay CPU (re, nhanh),
    nhung buoc nang (Hessian + tri rieng da ti le) chay tren GPU theo batch.
    Nhanh hon dang ke so voi ban CPU khi so anh lon hoac GPU ranh (Kaggle
    thuong co san GPU T4/P100 trong phien training)."""
    import os
    import cv2
    from tqdm import tqdm

    if cache_path is not None and os.path.exists(cache_path):
        print(f"[FRANGI CACHE] Tai cache co san tu {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print(f"[FRANGI-GPU] Tinh truoc vesselness map cho {len(img_ids)} anh "
          f"(batch_size={batch_size}, device={device})...")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frangi_maps = {}
    valid_ids = []
    tensors = []

    for img_id in img_ids:
        img_path = os.path.join(img_dir, img_id + img_ext)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [CANH BAO] khong doc duoc {img_path}, bo qua.")
            continue
        green = clahe.apply(img[:, :, 1])
        green = cv2.resize(green, (input_w, input_h))
        t = torch.from_numpy(green.astype(np.float32) / 255.0).unsqueeze(0)
        tensors.append(t)
        valid_ids.append(img_id)

    for i in tqdm(range(0, len(tensors), batch_size), desc="[FRANGI-GPU] Dang tinh (batch)"):
        batch_ids = valid_ids[i:i + batch_size]
        batch_tensor = torch.stack(tensors[i:i + batch_size]).to(device)  # (B,1,H,W)
        with torch.no_grad():
            vmaps = compute_frangi_map_gpu(batch_tensor, scale_range=(2, 8), device=device)
        vmaps = vmaps.squeeze(1).cpu().numpy()
        for j, img_id in enumerate(batch_ids):
            frangi_maps[img_id] = vmaps[j].astype(np.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.savez(cache_path, **frangi_maps)
        print(f"[FRANGI CACHE] Da luu cache vao {cache_path}")

    return frangi_maps


def frangi_map_to_tensor(vesselness: np.ndarray, device='cuda') -> torch.Tensor:
    """Chuyen numpy map (H,W) thanh tensor (1,1,H,W) de dung chung voi pipeline torch."""
    t = torch.from_numpy(vesselness).float().unsqueeze(0).unsqueeze(0)
    return t.to(device)


def precompute_frangi_maps(img_dir: str, img_ids: list, img_ext: str,
                            input_h: int, input_w: int, cache_path: str = None):
    """Tinh truoc Frangi map cho TOAN BO anh trong tap train, MOT LAN DUY NHAT
    truoc khi vao vong lap training - vi Frangi KHONG hoc, KHONG doi qua
    epoch, tinh lai moi batch la lang phi thoi gian khong can thiet.

    Luu ket qua vao dict {img_id: vesselness_map (H,W) da resize dung
    input_h x input_w de khop voi pipeline dataset chinh}, co the luu ra
    file .npz (cache_path) de tai lai nhanh cho lan chay sau, tranh tinh
    lai tu dau moi lan chay Kaggle session moi.
    """
    import os
    import cv2

    if cache_path is not None and os.path.exists(cache_path):
        print(f"[FRANGI CACHE] Tai cache co san tu {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print(f"[FRANGI] Tinh truoc vesselness map cho {len(img_ids)} anh "
          f"(chi 1 lan, khong can gradient)...")
    from tqdm import tqdm
    frangi_maps = {}
    for img_id in tqdm(img_ids, desc="[FRANGI] Dang tinh"):
        img_path = os.path.join(img_dir, img_id + img_ext)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [CANH BAO] khong doc duoc {img_path}, bo qua.")
            continue
        green = img[:, :, 1]
        vmap = compute_frangi_map(green, scale_range=(2, 8))
        vmap = cv2.resize(vmap, (input_w, input_h))
        frangi_maps[img_id] = vmap.astype(np.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.savez(cache_path, **frangi_maps)
        print(f"[FRANGI CACHE] Da luu cache vao {cache_path}")

    return frangi_maps


def frangi_assisted_merge(pred_region: torch.Tensor, pred_topology: torch.Tensor,
                           entropy_region: torch.Tensor, frangi_map: torch.Tensor,
                           lambda_thresh=(0.3, 0.5),
                           frangi_confidence_thresh=FRANGI_BINARY_THRESHOLD_DEFAULT,
                           num_iter=10):
    """Mo rong selective_merge_pseudo_label: them Frangi lam "trong tai thu 3"
    o vung Teacher_R (region) nghi ngo, khong chi dua vao Teacher_T (topology).

    frangi_confidence_thresh MAC DINH = 0.005 (FRANGI_BINARY_THRESHOLD_DEFAULT)
    - da kiem chung thuc te tren RITE, Dice trung binh 0.535 (n=5 anh,
      threshold CO DINH, khong do rieng tung anh). CAN kiem tra lai cho
      CHASE/HRF truoc khi dung chung gia tri nay.

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
