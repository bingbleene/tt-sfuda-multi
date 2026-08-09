#!/bin/bash
# run_resolution_pilot.sh
# =========================
# Chay TOAN BO pipeline kiem dinh y tuong resolution trong 1 lenh.
# Dat file nay vao TT_SFUDA_2D/ tren Kaggle roi chay:  !bash run_resolution_pilot.sh
#
# LAM GI:
#   1. Tao stage1 checkpoint cho CHASE, RITE (bo qua neu da co - idempotent)
#   2. Chay validate_resolution_scaling.py tren ca 3 dataset (khong train,
#      ~5-10 phut) - CHUA sua config gi, chi de xem
#   3. Sua input_h/input_w -> 1024 trong config CHASE->HRF (backup file cu)
#   4. Train THAT 3 seed (42/123/999) bang dung pipeline Dual-EMA+rollback
#      da kiem chung (~15-20 phut/seed o 1024px, uoc tinh)
#   5. In bang so sanh cuoi cung voi baseline

set -e  # dung ngay neu buoc nao loi, khong chay tiep buoc sau trong tinh trang sai

echo "=================================================================="
echo "BUOC 1/4: Tao stage1 checkpoint cho CHASE va RITE (neu chua co)"
echo "=================================================================="
if [ ! -f cache/stage1_hrf_chase.pth ]; then
    python run.py tt_sfuda_2d_region_topology.py --source hrf_unet --target chase \
        --stage1_ckpt cache/stage1_hrf_chase.pth --stage2_epochs 1 --lambda_cl 0 \
        --results_csv results_dummy_ignore.csv
else
    echo "  [BO QUA] cache/stage1_hrf_chase.pth da co san"
fi

if [ ! -f cache/stage1_chase_rite.pth ]; then
    python run.py tt_sfuda_2d_region_topology.py --source chase_unet --target rite \
        --stage1_ckpt cache/stage1_chase_rite.pth --stage2_epochs 1 --lambda_cl 0 \
        --results_csv results_dummy_ignore.csv
else
    echo "  [BO QUA] cache/stage1_chase_rite.pth da co san"
fi

echo ""
echo "=================================================================="
echo "BUOC 2/4: Kiem tra resolution scaling tren CA 3 dataset (khong train)"
echo "=================================================================="
echo "--- HRF (da chay truoc, chay lai de doi chieu) ---"
python run.py validate_resolution_scaling.py --source chase_unet --target hrf \
    --checkpoint cache/stage1_chase_hrf.pth --n_images 5

echo "--- CHASE ---"
python run.py validate_resolution_scaling.py --source hrf_unet --target chase \
    --checkpoint cache/stage1_hrf_chase.pth --n_images 5

echo "--- RITE ---"
python run.py validate_resolution_scaling.py --source chase_unet --target rite \
    --checkpoint cache/stage1_chase_rite.pth --n_images 5

echo ""
echo "=================================================================="
echo "BUOC 3/4: Sua config CHASE->HRF sang 1024x1024 (backup file cu truoc)"
echo "=================================================================="
cp models/chase_unet/config_hrf_dualema.yml models/chase_unet/config_hrf_dualema.yml.bak_512
python3 -c "
import yaml
path = 'models/chase_unet/config_hrf_dualema.yml'
with open(path) as f:
    c = yaml.safe_load(f)
c['input_h'] = 1024
c['input_w'] = 1024
with open(path, 'w') as f:
    yaml.dump(c, f)
print('Da sua input_h/input_w -> 1024x1024. Ban goc da luu tai',
      'models/chase_unet/config_hrf_dualema.yml.bak_512')
"

echo ""
echo "=================================================================="
echo "BUOC 4/4: Train THAT 3 seed o 1024px (dung pipeline Dual-EMA+rollback"
echo "da kiem chung 58.20+-0.09 - CHI doi resolution, khong doi gi khac)"
echo "=================================================================="
for SEED in 42 123 999; do
    echo "--- Seed $SEED ---"
    python run.py tt_sfuda_2d_dualema.py --source chase_unet --target hrf \
        --topology parallel --slow_keep_rate 0.995 \
        --early_stop_unsupervised --stage2_epochs 15 --seed $SEED \
        --results_csv results_resolution1024_pilot.csv
done

echo ""
echo "=================================================================="
echo "XONG. Chay tiep de doc ket qua:"
echo "  python run.py compare_resolution_pilot.py"
echo "=================================================================="
