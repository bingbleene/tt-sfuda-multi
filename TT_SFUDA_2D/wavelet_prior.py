"""
wavelet_prior.py
=================
Hướng (d): Wavelet-guided domain-invariant vesselness prior.

BỐI CẢNH / VÌ SAO KHÔNG LẶP LẠI THẤT BẠI CỦA FRANGI
-----------------------------------------------------
Frangi filter (đã thử, thất bại khi ép cứng xác suất 0.9 ở threshold 0.005:
-0.83 điểm Dice trên 3/3 seed) hoạt động trên KHÔNG GIAN (spatial domain) đơn
thuần: phân tích trị riêng Hessian đa tỉ lệ. Nhược điểm: threshold cố định
trên toàn ảnh không phân biệt được "vùng mạch máu thật mảnh, độ tương phản
thấp" với "nhiễu có dạng ống cục bộ" (mạch máu giả, nhiễu mao mạch, vệt sáng).

Wavelet decomposition khác về bản chất: nó định vị đồng thời CẢ không gian
VÀ tần số/hướng (horizontal/vertical/diagonal detail subbands), nên phân biệt
tốt hơn "cạnh có hướng nhất quán qua nhiều tỉ lệ" (đặc trưng mạch máu thật -
một đường mảnh sẽ có năng lượng nhất quán ở same location qua level 1,2,3)
với nhiễu ngẫu nhiên (năng lượng KHÔNG nhất quán qua các level). Đây chính là
tín hiệu ta dùng làm "wavelet vesselness": consistency của energy qua các
level, không phải chỉ giá trị tuyệt đối ở 1 tỉ lệ như Frangi.

BÀI HỌC ÁP DỤNG TỪ THẤT BẠI FRANGI
-----------------------------------
1. KHÔNG ép cứng xác suất (hard override) - chỉ dùng làm vote/weight bổ
   sung, kết hợp mềm với pseudo-label hiện có.
2. Threshold phải được calibrate theo per-image percentile, KHÔNG dùng
   ngưỡng tuyệt đối cố định toàn dataset (bài học: 0.005 quá lỏng khi ép
   cứng, nhưng vấn đề cốt lõi hơn là ngưỡng tuyệt đối không ổn định qua các
   domain shift khác nhau - ảnh HRF sáng/tương phản khác CHASE/RITE).
3. Validate ĐỘC LẬP trước (so trực tiếp với GT, không qua pipeline) như đã
   làm với Frangi (Dice ~0.44-0.53), TRƯỚC KHI tích hợp vào Dual-EMA.

Dependency: pywavelets (pip install PyWavelets --break-system-packages)
"""

import numpy as np

try:
    import pywt
except ImportError as e:
    raise ImportError(
        "Cần cài PyWavelets trước: pip install PyWavelets --break-system-packages"
    ) from e


def _energy_map(coeffs_2d: np.ndarray, target_shape):
    """Upsample magnitude map của 1 subband về kích thước ảnh gốc."""
    from scipy.ndimage import zoom
    zy = target_shape[0] / coeffs_2d.shape[0]
    zx = target_shape[1] / coeffs_2d.shape[1]
    return zoom(np.abs(coeffs_2d), (zy, zx), order=1)


def wavelet_vesselness(image_gray: np.ndarray, wavelet: str = "db4",
                        levels: int = 3, percentile_threshold: float = 92.0):
    """
    image_gray: ảnh grayscale (đã qua CLAHE giống pipeline Frangi hiện có
                của bạn), float trong [0,1], shape (H, W).
    wavelet:    loại wavelet, 'db4' hoặc 'sym4' đều phù hợp cho cạnh mảnh.
    levels:     số mức phân giải (2-3 là đủ cho mạch máu 1-3px ở scale gốc,
                đừng tăng quá cao vì mạch máu sẽ bị "hoà tan" vào tỉ lệ thô).
    percentile_threshold: dùng percentile THEO TỪNG ẢNH (không phải ngưỡng
                tuyệt đối cố định) để tính consistency - đây là điểm sửa lỗi
                so với Frangi.

    Trả về: wavelet_vesselness_map, giá trị liên tục trong [0,1], KHÔNG
            threshold cứng - để nơi gọi hàm này tự quyết định cách dùng làm
            vote (khuyến nghị: dùng làm weight nhân vào agreement_fusion
            của cross_arch_teacher.py, không override trực tiếp pseudo-label).
    """
    H, W = image_gray.shape
    coeffs = pywt.wavedec2(image_gray, wavelet=wavelet, level=levels)
    # coeffs[0] = cA (approximation, level thô nhất) -> bỏ qua, không mang
    # thông tin cạnh mảnh.
    detail_levels = coeffs[1:]  # list of (cH, cV, cD) từ thô -> mịn

    per_level_maps = []
    for (cH, cV, cD) in detail_levels:
        # gộp 3 hướng (horizontal/vertical/diagonal) thành 1 "orientation
        # energy" - mạch máu có thể chạy theo bất kỳ hướng nào nên không
        # chọn riêng 1 hướng như một số phương pháp domain-generalization
        # vessel khác.
        combined = np.sqrt(cH.astype(np.float64) ** 2 +
                            cV.astype(np.float64) ** 2 +
                            cD.astype(np.float64) ** 2)
        per_level_maps.append(_energy_map(combined, (H, W)))

    # Chuẩn hoá mỗi level về [0,1] theo percentile CỦA CHÍNH ẢNH ĐÓ (per-image
    # calibration - sửa lỗi threshold tuyệt đối của Frangi).
    normalized = []
    for m in per_level_maps:
        thresh_val = np.percentile(m, percentile_threshold)
        thresh_val = max(thresh_val, 1e-8)
        normalized.append(np.clip(m / thresh_val, 0, 1))

    # "Consistency" qua các level = tín hiệu chính, thay vì chỉ 1 tỉ lệ.
    # Nhân theo phần tử (không cộng trung bình): một điểm chỉ được coi là
    # mạch máu thật nếu NHẤT QUÁN có năng lượng cao ở NHIỀU level - đây là
    # điểm khác biệt cốt lõi so với Frangi (vốn dùng max qua các scale, dễ
    # bắt nhiễu chỉ mạnh ở 1 scale).
    consistency = np.ones((H, W), dtype=np.float64)
    for m in normalized:
        consistency *= m

    # consistency đã tự nhiên nằm trong [0,1] vì mỗi m trong [0,1].
    return consistency.astype(np.float32)


def wavelet_vote_weight(image_gray: np.ndarray, existing_pseudo_prob: np.ndarray,
                         alpha: float = 0.3, **wavelet_kwargs):
    """
    Kết hợp MỀM wavelet vesselness vào pseudo-label hiện có, thay vì ép cứng
    (bài học từ Frangi: ép cứng ở threshold lỏng làm giảm Dice 0.83 điểm).

    existing_pseudo_prob: pseudo-label xác suất hiện tại của bạn (từ
                           Dual-Teacher agreement_fusion hoặc pipeline cũ).
    alpha:  trọng số đóng góp của wavelet prior, mặc định 0.3 (nhỏ, thận
            trọng - tăng dần sau khi validate độc lập, đừng bắt đầu bằng
            alpha cao như 0.9 đã làm với Frangi).

    Trả về pseudo-label đã kết hợp, VẪN trong [0,1], liên tục (không có
    bước "đẩy mạnh xác suất" cứng nhắc nào).
    """
    wv = wavelet_vesselness(image_gray, **wavelet_kwargs)
    combined = (1 - alpha) * existing_pseudo_prob + alpha * wv
    return np.clip(combined, 0, 1)


# ---------------------------------------------------------------------------
# Validate ĐỘC LẬP trước khi tích hợp - lặp lại đúng quy trình đã làm với
# Frangi (so trực tiếp Dice với GT trên vài ảnh mỗi dataset).
# ---------------------------------------------------------------------------

def evaluate_standalone_dice(pred_prob: np.ndarray, gt_binary: np.ndarray,
                              threshold: float = 0.5) -> float:
    pred_binary = (pred_prob >= threshold).astype(np.uint8)
    gt_binary = gt_binary.astype(np.uint8)
    intersection = (pred_binary & gt_binary).sum()
    denom = pred_binary.sum() + gt_binary.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom
