"""
geometry_prior.py
====================
Module DOC LAP, khong phu thuoc V3/V4/V5b - tinh "geometry prior" S_geometry(r)
de cong vao S_feature(r) cua V5b theo cong thuc da thong nhat:

    S(r) = S_feature(r) + lambda * S_geometry(r)

Y NGHIA CUA source_calib_ratio (lam ro lai, tranh hieu nham la "biet do rong
vessel that"): day CHI la 1 THUC TE VE PHEP RESIZE - ty le anh nguon bi co
lai luc train nguon (512 / kich thuoc native trung binh cua chinh dataset
nguon). KHONG tuyen bo biet noi dung/FOV/camera/vessel distribution that
cua target - chi tuyen bo: "day la muc do co-rut ma chinh source model da
quen chiu dung", roi so target o moi candidate resolution co dang co-rut
"gan" hay "xa" muc do do. Yeu hon 1 proxy that ve vessel scale, nhung KHONG
dung anh/nhan nguon - chi dung 1 con so metadata (kich thuoc anh) da biet
tu truoc, ghi 1 lan, khong doc lai luc chon cau hinh.

NATIVE_SIZES duoi day da tinh san (trung binh H,W tren vai anh moi dataset,
qua check_resolution_gap.py truoc do) - KHONG doc lai anh nguon luc chay
selector, chi dung hang so da ghi nhan san (giong nhu biet learning rate
da dung, khong phai truy cap lai du lieu train).

Cach dung (sau khi V5b tinh xong S_feature(r) cho tung transition):
    from geometry_prior import geometry_penalty, source_calib_ratio

    penalty_small = geometry_penalty(res_small, target, source)
    penalty_large = geometry_penalty(res_large, target, source)
    geometry_term = penalty_small - penalty_large  # duong neu di len GIAM penalty
    combined_utility = feature_utility + LAMBDA * geometry_term
"""

# Kich thuoc native trung binh (H, W) -> luu gon thanh 1 so trung binh don gian,
# da tinh tu check_resolution_gap.py chay truoc do trong du an.
NATIVE_SIZE_AVG = {
    'chase': 979.5,   # (960 + 999) / 2
    'hrf': 2920.0,    # (2336 + 3504) / 2
    'rite': 574.5,    # (584 + 565) / 2
}

# Moi source model duoc train goc o 512x512 tren chinh dataset nguon cua no
# (quy uoc co san trong toan bo du an - khong doi).
SOURCE_TRAIN_RESOLUTION = 512

# Ten model -> ten dataset native tuong ung (de tra NATIVE_SIZE_AVG dung)
MODEL_TO_DATASET = {
    'chase_unet': 'chase',
    'hrf_unet': 'hrf',
    'rite_unet': 'rite',
}


def source_calib_ratio(source_model_name: str) -> float:
    """Ty le co-rut ma source model DA QUEN luc train goc.
    vd hrf_unet: 512 / 2920 = 0.175 (co rut RAT MANH - HRF native cuc lon)."""
    dataset = MODEL_TO_DATASET[source_model_name]
    native = NATIVE_SIZE_AVG[dataset]
    return SOURCE_TRAIN_RESOLUTION / native


def target_ratio_at_resolution(resolution: int, target_dataset_name: str) -> float:
    """Ty le co-rut cua TARGET neu resize ve `resolution`."""
    native = NATIVE_SIZE_AVG[target_dataset_name]
    return resolution / native


def geometry_penalty(resolution: int, target_dataset_name: str, source_model_name: str) -> float:
    """Khoang cach tuyet doi giua ty le co-rut cua target (o resolution nay)
    va ty le ma source da quen. CANG LON = resolution nay cang "la" so voi
    nhung gi source model tung thay luc train - CANG DANG NGHI NGO."""
    calib = source_calib_ratio(source_model_name)
    target_r = target_ratio_at_resolution(resolution, target_dataset_name)
    return abs(target_r - calib)


def combined_marginal_utility(feature_utility: float, res_small: int, res_large: int,
                                target_dataset_name: str, source_model_name: str,
                                lam: float) -> float:
    """Cong thuc da thong nhat: S(r) = S_feature(r) + lambda * S_geometry(r).

    O day the hien duoi dang MARGINAL (khop voi phong cach V4): feature_utility
    la utility tu V5b cho 1 buoc chuyen res_small->res_large; geometry_term la
    THAY DOI penalty hinh hoc khi di tu res_small len res_large - AM neu penalty
    tang (cang xa calib nguon hon), lam GIAM utility tong - dung dinh huong
    "phat" viec di xa khoi vung nguon quen thuoc.
    """
    penalty_small = geometry_penalty(res_small, target_dataset_name, source_model_name)
    penalty_large = geometry_penalty(res_large, target_dataset_name, source_model_name)
    geometry_term = penalty_small - penalty_large  # duong neu di len GIAM duoc penalty
    return feature_utility + lam * geometry_term


def sweep_lambda_table(feature_utilities: dict, target_dataset_name: str,
                         source_model_name: str, lambdas=(0.0, 0.1, 0.3, 0.5, 1.0)):
    """Ho tro ablation lambda de xuat truoc - dua vao feature_utilities da tinh
    san (dict {(res_small,res_large): feature_utility_value} tu V5b), in bang
    combined utility qua nhieu lambda de xem lambda nao thay doi quyet dinh
    dung/sai so voi Dice that da biet.
    """
    print(f"{'Transition':<16}", end='')
    for lam in lambdas:
        print(f"lambda={lam:<10}", end='')
    print()

    for (res_small, res_large), feat_u in feature_utilities.items():
        print(f"{res_small}->{res_large:<10}", end='')
        for lam in lambdas:
            combined = combined_marginal_utility(feat_u, res_small, res_large,
                                                   target_dataset_name, source_model_name, lam)
            print(f"{combined:<16.4f}", end='')
        print()


if __name__ == '__main__':
    # Demo / xac minh nhanh - khop dung bang da tinh tay truoc do khi thao luan
    print("=== source_calib_ratio cho tung source model ===")
    for m in MODEL_TO_DATASET:
        print(f"  {m}: {source_calib_ratio(m):.4f}")

    print("\n=== geometry_penalty(resolution) cho ca 4 domain shift ===")
    shifts = [('chase_unet', 'hrf'), ('chase_unet', 'rite'),
              ('hrf_unet', 'chase'), ('hrf_unet', 'rite')]
    for source, target in shifts:
        print(f"\n{source} -> {target} (calib={source_calib_ratio(source):.3f}):")
        for res in [384, 512, 768, 1024]:
            p = geometry_penalty(res, target, source)
            print(f"    {res}px: penalty={p:.4f}")
