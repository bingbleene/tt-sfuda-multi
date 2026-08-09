"""
diagnose_cross_arch_divergence.py
====================================
PILOT cho huong (a) Cross-architecture co-training - chay TRUOC khi cam ket
tt_sfuda_2d_cross_arch.py full (15 epoch x nhieu seed x 4 domain shift).

Bai hoc ap dung: lan thu Region/Topology Teacher truoc day chi phat hien
2 teacher gan nhu giong het nhau (~0.002-0.003 chenh lech) SAU KHI da chay
het pipeline. Script nay lam dung 1 viec: train warmup NGAN (vai epoch),
roi do do phan ky - KHONG chay full 15 epoch, KHONG ghi vao results_csv
chinh, chi in ra verdict GO / KHONG GO de quyet dinh buoc tiep theo.

Yeu cau: da co stage1 checkpoint san (dung --stage1_ckpt tu lan chay
tt_sfuda_2d_region_topology.py hoac tt_sfuda_2d_dualema.py truoc do, de
khong phai train lai Stage I).

Cach dung:
    python run.py diagnose_cross_arch_divergence.py --source chase_unet \\
        --target hrf --stage1_ckpt cache/stage1_chase_hrf.pth \\
        --warmup_epochs 2
"""
import os
import argparse
from glob import glob

import yaml
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from albumentations import RandomRotate90, Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import archs
import losses
from dataset import Dataset

from cross_arch_bottleneck import UNetDilatedBottleneck, load_from_unet_checkpoint
from region_topology_teacher import selective_merge_pseudo_label, soft_skeletonize
from tt_sfuda_2d_region_topology import train_epoch, entropy_map, ema_update

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--stage1_ckpt', required=True,
                         help='Checkpoint Stage I co san (BAT BUOC - script nay khong tu train Stage I).')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                         help='So epoch warmup NGAN truoc khi do phan ky. 2 la diem khoi dau hop ly '
                              '- neu can, tang len 3-4 va chay lai, dung nhay thang len 15.')
    parser.add_argument('--ema_keep_rate', type=float, default=0.99)
    parser.add_argument('--merge_lambda1', type=float, default=0.3)
    parser.add_argument('--merge_lambda2', type=float, default=0.5)
    parser.add_argument('--disagree_threshold', type=float, default=0.3,
                         help='Nguong chenh lech xac suat (|p_r - p_x|) de tinh la 1 pixel bat dong.')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_model(config):
    m = archs.__dict__[config['arch']](config['num_classes'],
                                        config['input_channels'],
                                        config['deep_supervision'])
    return m.cuda()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    config_file = "config_" + args.target + "_dualema"
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_img_ids = glob(os.path.join('inputs', args.target, 'train', 'images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    train_transform = Compose([
        RandomRotate90(), transforms.Flip(),
        Resize(config['input_h'], config['input_w']), transforms.Normalize(),
    ])
    train_dataset = Dataset(
        img_ids=train_img_ids, img_dir=os.path.join('inputs', args.target, 'train', 'images'),
        mask_dir=os.path.join('inputs', args.target, 'train', 'masks'),
        img_ext=config['img_ext'], mask_ext=config['mask_ext'],
        num_classes=config['num_classes'], transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=config['num_workers'], drop_last=True)

    assert os.path.exists(args.stage1_ckpt), (
        f"Khong tim thay stage1_ckpt tai {args.stage1_ckpt}. Script nay CAN checkpoint co san, "
        f"chay tt_sfuda_2d_region_topology.py --stage1_ckpt <path> mot lan truoc de tao no.")

    print(f"[SETUP] Nap Stage I checkpoint tu {args.stage1_ckpt}")

    # --- Teacher_R / Student_R: UNet chuan, y het pipeline goc ---
    student_r = build_model(config)
    student_r.load_state_dict(torch.load(args.stage1_ckpt))
    teacher_r = build_model(config)
    teacher_r.load_state_dict(torch.load(args.stage1_ckpt))

    # --- Teacher_X / Student_X: cross-architecture (ASPP bottleneck) ---
    student_x = UNetDilatedBottleneck(config['num_classes'], config['input_channels'],
                                       config['deep_supervision']).cuda()
    load_from_unet_checkpoint(student_x, args.stage1_ckpt)
    teacher_x = UNetDilatedBottleneck(config['num_classes'], config['input_channels'],
                                       config['deep_supervision']).cuda()
    load_from_unet_checkpoint(teacher_x, args.stage1_ckpt)

    opt_r = optim.Adam(student_r.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    opt_x = optim.Adam(student_x.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    criterion = losses.__dict__[config['loss']]().cuda()
    merge_kwargs = dict(lambda_thresh=(args.merge_lambda1, args.merge_lambda2))

    # --- Do phan ky NGAY TU DAU (epoch 0, truoc khi train buoc nao) ---
    print("\n[BUOC 0] Do phan ky NGAY SAU KHI KHOI TAO (chua train buoc nao)")
    print("         -> phan anh thuan tuy do khac biet KIEN TRUC, chua bi anh huong boi hoc.")
    _report_divergence(teacher_r, teacher_x, train_loader, args.disagree_threshold, tag="epoch 0 (khoi tao)")

    # --- Warmup NGAN, dung LAI train_epoch da co san (khong viet lai) ---
    print(f"\n[BUOC 1] Warmup {args.warmup_epochs} epoch (dung chung train_epoch() "
          f"tu tt_sfuda_2d_region_topology.py, lambda_cl=0 de tat clDice - "
          f"chi test THUAN kien truc, chua tron them yeu to khac).")
    for ep in range(args.warmup_epochs):
        log = train_epoch(train_loader, student_r, teacher_r, opt_r,
                           student_x, teacher_x, opt_x,
                           criterion, cldice_loss=None, lambda_cl=0.0,
                           ema_keep_rate=args.ema_keep_rate, merge_kwargs=merge_kwargs,
                           frangi_maps=None)
        print(f"  warmup epoch {ep + 1}/{args.warmup_epochs} - loss_r {log['loss_r']:.4f} "
              f"- loss_x {log['loss_x'] if 'loss_x' in log else log.get('loss_t', float('nan')):.4f}")
        _report_divergence(teacher_r, teacher_x, train_loader, args.disagree_threshold,
                            tag=f"sau warmup epoch {ep + 1}")

    print("\n" + "=" * 70)
    print("KET LUAN: xem so 'global_disagreement_rate' o BUOC cuoi cung ben tren.")
    print("  >= 0.05 (5%)  -> KIEN TRUC DA DU PHAN KY. Co the chay tt_sfuda_2d_cross_arch.py")
    print("                   full (15 epoch, nhieu seed) voi cau hinh nay.")
    print("  0.02 - 0.05   -> Ranh gioi. Thu tang so dilation ASPP hoac chay them 1-2 warmup")
    print("                   epoch nua truoc khi ket luan.")
    print("  < 0.02 (2%)   -> CHUA DU PHAN KY - DUNG chay full pipeline. Tang dilation rate")
    print("                   trong cross_arch_bottleneck.py (vd (1,4,8,16)) roi do lai tu dau.")
    print("=" * 70)


@torch.no_grad()
def _report_divergence(teacher_r, teacher_x, loader, disagree_threshold, tag, max_batches=30):
    teacher_r.eval()
    teacher_x.eval()

    total_pixels, disagree_pixels = 0, 0
    total_relevant, disagree_relevant = 0, 0
    # Do them chenh lech O TUNG TANG FEATURE (x1_0..x4_0) - de biet chinh xac
    # sai khac bi "chet" o tang nao, thay vi chi thay output cuoi = 0 ma
    # khong ro nguyen nhan (giup debug nhanh hon neu v2 nay van chua du).
    level_names = ['x1_0', 'x2_0', 'x3_0', 'x4_0']
    level_diff_sum = [0.0] * 4
    level_norm_sum = [0.0] * 4
    n_batch = 0

    for i, (input, _, _) in enumerate(loader):
        if i >= max_batches:
            break
        input = input.cuda()
        out_r, feat_r = teacher_r(input, mode='const')
        out_x, feat_x = teacher_x(input, mode='const')
        prob_r = torch.sigmoid(out_r)
        prob_x = torch.sigmoid(out_x)
        diff = (prob_r - prob_x).abs()

        disagree = diff > disagree_threshold
        total_pixels += disagree.numel()
        disagree_pixels += disagree.sum().item()

        relevant = (prob_r > 0.1) | (prob_x > 0.1)
        total_relevant += relevant.sum().item()
        disagree_relevant += (disagree & relevant).sum().item()

        for lvl in range(4):
            fr, fx = feat_r[lvl], feat_x[lvl]
            level_diff_sum[lvl] += (fr - fx).abs().mean().item()
            level_norm_sum[lvl] += fr.abs().mean().item() + 1e-8
        n_batch += 1

    global_rate = disagree_pixels / max(total_pixels, 1)
    relevant_rate = disagree_relevant / max(total_relevant, 1)

    print(f"  [DIVERGENCE @ {tag}] global_disagreement_rate={global_rate:.4f} "
          f"({global_rate*100:.2f}%), disagreement_in_vessel_region={relevant_rate:.4f} "
          f"({relevant_rate*100:.2f}%) tren {min(max_batches, len(loader))} anh")

    level_report = ", ".join(
        f"{name}: {level_diff_sum[i]/n_batch:.6f} (tuong doi ~{100*level_diff_sum[i]/level_norm_sum[i]:.2f}%)"
        for i, name in enumerate(level_names))
    print(f"  [DIVERGENCE PER-LEVEL @ {tag}] {level_report}")
    print(f"    -> neu x1_0/x2_0/x3_0 gan 0 nhung x4_0 khac 0 ro: sai khac dang bi 'chet' "
          f"o cac tang decoder phia sau do skip connection. Can dat diem khac biet kien "
          f"truc o tang NONG hon (x2_0 hoac x1_0), khong chi bottleneck.")

    teacher_r.train()
    teacher_x.train()


if __name__ == '__main__':
    main()
