"""
sam_prior.py
=============
SAM (Segment Anything, VANILLA - KHONG fine-tune) lam nguon vote thu 3,
DOC LAP HOAN TOAN ve kien truc voi Teacher_R/Teacher_T: SAM la 1 mang
rieng biet chay ngoai pipeline, khong share layer/skip connection/gradient
nao voi UNet - day moi la kieu "khac biet cau truc that" ma
cross_arch_bottleneck.py (ASPP o bottleneck) co gang dat nhung THAT BAI vi
bi skip connection noi bo cua chinh UNet "nuot" mat (xem ket qua
diagnose_cross_arch_divergence.py - disagreement=0.0000 du feature noi bo
lech >1000%).

CANH BAO QUAN TRONG (dua tren literature da khao sat, doc truoc khi chay):
- TAT CA paper 2024-2025 dung SAM cho fundus/vessel (SAM-OSLN MICCAI2025,
  IPLC MICCAI2024, IPLC+ JBHI2025, SRPL-SFDA 2025) DEU fine-tune SAM truoc
  khi dung, KHONG ai dung zero-shot thuan. SAM duoc train de nhan dien
  "vat the" co ranh gioi ro trong anh tu nhien - mach mau la cau truc dang
  day/luoi MANH (1-3px), la diem yeu da duoc chinh tac gia SAM-OSLN thua
  nhan ngay ca SAU KHI fine-tune ("neighbor-voting strategy does miss some
  tiny vessels").
- Script nay VAN validate DOC LAP truoc (dung dung quy trinh da lam voi
  Frangi va wavelet) de co so lieu THAT. Ky vong THAP la hop ly - neu Dice
  doc lap thua Frangi (0.4678), dong huong nay giong wavelet, KHONG co
  gang fine-tune SAM (ton thoi gian, ngoai pham vi 1 pilot nhanh).

Phu thuoc:
  pip install git+https://github.com/facebookresearch/segment-anything.git --break-system-packages
Checkpoint (ViT-B, nho nhat, ~375MB, du cho pilot):
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
"""
import numpy as np


def build_sam_predictor(checkpoint_path: str, model_type: str = 'vit_b', device: str = 'cuda'):
    """Khoi tao 1 lan, dung lai cho nhieu anh (set_image() moi lan doi anh,
    KHONG can khoi tao lai predictor - tranh load checkpoint nhieu lan)."""
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def _make_point_grid(h, w, points_per_side):
    ys = np.linspace(0, h - 1, points_per_side, dtype=int)
    xs = np.linspace(0, w - 1, points_per_side, dtype=int)
    return np.array([[x, y] for y in ys for x in xs])


def compute_sam_vesselness_map(predictor, image_rgb_uint8: np.ndarray,
                                points_per_side: int = 16,
                                min_score: float = 0.5) -> np.ndarray:
    """
    Tra ve map (H,W) trong [0,1] - do tin cay SAM tai tung pixel, tong hop
    tu nhieu diem prompt phan bo deu khap anh.

    CACH DUNG DON GIAN NHAT co the (khong phai automatic mask generator
    day du - qua cham cho 1 pilot; khong fine-tune). points_per_side=16
    -> 256 diem prompt, moi diem 1 lan goi predict() (nhanh sau khi
    set_image() da tinh embedding 1 lan cho ca anh).

    Voi moi diem, lay mask co score cao nhat trong 3 mask SAM tra ve, MAX
    (khong cong don) vao map chung - tranh vung duoc nhieu diem trung lap
    trung tam bi khuech dai gia tao.
    """
    h, w = image_rgb_uint8.shape[:2]
    predictor.set_image(image_rgb_uint8)

    grid = _make_point_grid(h, w, points_per_side)
    vmap = np.zeros((h, w), dtype=np.float32)

    for (x, y) in grid:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < min_score:
            continue
        vmap = np.maximum(vmap, masks[best_idx].astype(np.float32) * best_score)

    return np.clip(vmap, 0, 1)
