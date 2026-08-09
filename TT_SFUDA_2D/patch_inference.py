"""
patch_inference.py
=====================
Y TUONG "BUT PHA" - khac han moi thu da thu (khong phai architecture/loss/
predictor phu/hau xu ly), ma la SUA LAI CACH THONG TIN DI VAO MODEL.

CO SO: check_resolution_gap.py da xac nhan BANG SO LIEU THAT - anh resize
ve input_h x input_w co dinh (512x512) lam mach mau mong nhat con LAI:
  CHASE: 0.52px | HRF: 0.18px | RITE: 0.89px
O ca 3 dataset, mach mau mong nhat da o duoi hoac sat nguong 1 pixel -
thong tin MAT VINH VIEN truoc khi model kip thay, khong thuat toan nao
(Stage I/II, ensemble, hau xu ly) co the cuu lai duoc.

GIAI PHAP: thay vi resize CA ANH xuong nho, CAT PATCH (cua so nho) TU ANH
GOC O DO PHAN GIAI NATIVE (khong resize), chay tung patch qua model, GHEP
LAI (stitch) thanh du doan day du kich thuoc goc. Day la ky thuat CHUAN
trong xu ly anh y te do phan giai cao (WSI, CT/MRI full-res) - khong phai
y tuong moi la, nhung CHUA TUNG duoc ap dung trong du an nay (moi thu
nghiem truoc gio deu giu nguyen buoc resize input_h x input_w).

GIAI DOAN 1 (script validate_patch_inference.py): dung CHECKPOINT DA CO
SAN, KHONG TRAIN LAI GI - chi doi cach INFERENCE, so Dice voi cach resize
toan anh cu. Neu co cai thien ro rang du KHONG train lai cho patch, day la
bang chung rat manh (gan nhu chac chan) rang GIAI DOAN 2 (train lai voi du
lieu dang patch) se cho ket qua tot hon nhieu.
"""
import numpy as np
import cv2
import torch
from albumentations.augmentations import transforms

_albu_normalize = transforms.Normalize()


def normalize_like_dataset(image_bgr_uint8: np.ndarray) -> np.ndarray:
    """PHAI khop CHINH XAC voi dataset.py that: Compose transform (bao gom
    transforms.Normalize() albumentations) RoI COIN chia them /255 MOT LAN
    NUA sau do (dong 'img = img.astype("float32") / 255' trong dataset.py).
    Day la quirk co san trong pipeline goc - model DA DUOC TRAIN theo dung
    scale nay, nen suy luan phai lap lai y het, du nhin la "chuan hoa kep"
    la thua. BO QUA buoc nay se lam input sai lech ~255 lan so voi scale
    model quen -> bao hoa toan mang -> du doan vo nghia (day chinh la
    nguyen nhan Dice=0.0000 tuyet doi o ca 2 cach trong lan chay truoc).

    LUU Y: anh dau vao PHAI la BGR (cv2.imread mac dinh, KHONG chuyen RGB) -
    dataset.py khong goi cvtColor, giu nguyen BGR."""
    img = _albu_normalize(image=image_bgr_uint8)['image']
    img = img.astype('float32') / 255.0
    return img


def tile_positions(h, w, patch_h, patch_w, stride_h, stride_w):
    """Danh sach (y0, x0) top-left cua tung patch, PHU KIN toan bo anh
    (bao gom canh/goc, du kich thuoc anh khong chia het cho stride)."""
    ys = list(range(0, max(h - patch_h, 0) + 1, stride_h))
    if not ys or ys[-1] != max(h - patch_h, 0):
        ys.append(max(h - patch_h, 0))
    xs = list(range(0, max(w - patch_w, 0) + 1, stride_w))
    if not xs or xs[-1] != max(w - patch_w, 0):
        xs.append(max(w - patch_w, 0))
    return [(y, x) for y in ys for x in xs]


def _get_patch(image, y0, x0, patch_h, patch_w):
    h, w = image.shape[:2]
    y1, x1 = min(y0 + patch_h, h), min(x0 + patch_w, w)
    patch = image[y0:y1, x0:x1]
    valid_h, valid_w = patch.shape[0], patch.shape[1]
    pad_h, pad_w = patch_h - valid_h, patch_w - valid_w
    if pad_h > 0 or pad_w > 0:
        patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return patch, valid_h, valid_w


@torch.no_grad()
def patch_predict_native(model, image_native_rgb: np.ndarray, patch_size,
                          overlap_ratio: float = 0.5, device: str = 'cuda',
                          batch_size: int = 8) -> np.ndarray:
    """
    Tra ve xac suat (H,W) O DUNG DO PHAN GIAI NATIVE cua image_native_rgb -
    KHONG resize toan anh, chi cat patch kich thuoc patch_size (khop dung
    input_h/input_w model da train), chay tung patch, GHEP LAI bang trung
    binh co trong so o vung chong lan (overlap) de tranh vien ranh gioi.

    model: da .eval(), nhan input (B,3,patch_h,patch_w) tra ve logits
           (B,1,patch_h,patch_w) - dung interface archs.UNet hien co.
    patch_size: (patch_h, patch_w) - NEN dung dung (config['input_h'],
                config['input_w']) de patch dung dung kich thuoc model
                da quen trong luc train (khong doi statistics dau vao).
    overlap_ratio: 0.5 = stride bang 1/2 patch (chong lan 50%, lam muot
                   ranh gioi giua cac patch). Tang len (vd 0.75) cho ket
                   qua muot hon nhung CHAM hon nhieu - 0.5 la diem khoi
                   dau hop ly cho 1 pilot nhanh.
    """
    patch_h, patch_w = patch_size
    h, w = image_native_rgb.shape[:2]
    stride_h = max(1, int(patch_h * (1 - overlap_ratio)))
    stride_w = max(1, int(patch_w * (1 - overlap_ratio)))

    positions = tile_positions(h, w, patch_h, patch_w, stride_h, stride_w)

    prob_accum = np.zeros((h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    for i in range(0, len(positions), batch_size):
        batch_positions = positions[i:i + batch_size]
        batch_patches, batch_valid = [], []
        for (y0, x0) in batch_positions:
            patch, vh, vw = _get_patch(image_native_rgb, y0, x0, patch_h, patch_w)
            patch_norm = normalize_like_dataset(patch)
            batch_patches.append(patch_norm.transpose(2, 0, 1))
            batch_valid.append((y0, x0, vh, vw))

        batch_tensor = torch.from_numpy(np.stack(batch_patches)).float().to(device)
        output = model(batch_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[:, 0]  # (B, patch_h, patch_w)

        for j, (y0, x0, vh, vw) in enumerate(batch_valid):
            prob_accum[y0:y0 + vh, x0:x0 + vw] += probs[j, :vh, :vw]
            weight_accum[y0:y0 + vh, x0:x0 + vw] += 1.0

    weight_accum = np.clip(weight_accum, 1e-6, None)
    return prob_accum / weight_accum
