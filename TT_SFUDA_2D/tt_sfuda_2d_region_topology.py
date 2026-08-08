"""
tt_sfuda_2d_region_topology.py
================================
Kien truc Region-Teacher / Topology-Teacher: 2 CAP student-teacher DOC LAP
HOAN TOAN (khac optimizer, khac lan backward), khac voi ban
tt_sfuda_2d_control_cldice.py truoc day (1 model, CONG DON 2 loss vao
CUNG 1 lan backward - da xac nhan gay xung dot gradient, train_loss tang
vot 2.4-3.5, ket qua te hon ca khong co class-balance).

Nguyen tac thiet ke (dua tren DBTS 2024 va Ensemble-Distillation CVPRW 2021,
xem giai thich trong hoi thoai):
  - Pair R (Region):   Student_R + Teacher_R (EMA cua Student_R).
                        Loss = Dice+BCE thuong (giong criterion goc).
  - Pair T (Topology):  Student_T + Teacher_T (EMA cua Student_T).
                        Loss = Dice+BCE + lambda_cl * clDice.
  - HAI optimizer TACH BIET, HAI lan .backward() TACH BIET moi batch -
    khong co xung dot gradient giua 2 muc tieu.
  - Pseudo-label DUNG CHUNG cho ca 2 student: selective_merge_pseudo_label(
    Teacher_R, Teacher_T, entropy_R) - Teacher_R lam nen, Teacher_T va vung
    nghi ngo dut doan (xem region_topology_teacher.py).
  - EMA rate: CA HAI teacher dung CUNG 1 keep_rate (mac dinh 0.99, giong
    ban dualema goc) - KHONG con la Fast/Slow nua, vi bien so "khac nhau"
    o day la LOSS MUC TIEU, khong phai toc do EMA.

Cach dung (LUON qua run.py, giong moi script khac trong repo):
    python run.py tt_sfuda_2d_region_topology.py --source chase_unet \
        --target hrf --lambda_cl 1.0 --stage2_epochs 15 \
        --results_csv results_region_topology.csv
"""
import os
import copy
import csv
import random
import yaml
import argparse
from datetime import datetime
from glob import glob
from tqdm import tqdm
from collections import OrderedDict, deque

import numpy as np
from albumentations import RandomRotate90, Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter

from tt_sfuda_2d import (
    build_strong_augmentation,
    consistency_loss,
    sfuda_target,
    validate,
)
from region_topology_teacher import SoftClDiceLoss, selective_merge_pseudo_label
from frangi_prior import (
    precompute_frangi_maps,
    frangi_assisted_merge,
    FRANGI_BINARY_THRESHOLD_DEFAULT,
)

cudnn.benchmark = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage1_ckpt', default=None)
    parser.add_argument('--stage2_epochs', type=int, default=None)
    parser.add_argument('--ema_keep_rate', type=float, default=0.99)
    parser.add_argument('--lambda_cl', type=float, default=1.0)
    parser.add_argument('--cldice_num_iter', type=int, default=10)
    parser.add_argument('--merge_lambda1', type=float, default=0.3)
    parser.add_argument('--merge_lambda2', type=float, default=0.5)
    parser.add_argument('--use_frangi', action='store_true',
                         help='Bat Frangi vesselness lam "trong tai thu 3" trong buoc '
                              'gop pseudo-label. Da kiem chung tren 15 anh (CHASE/HRF/RITE), '
                              'Dice trung binh 0.4678, threshold co dinh 0.005.')
    parser.add_argument('--frangi_threshold', type=float, default=FRANGI_BINARY_THRESHOLD_DEFAULT)
    parser.add_argument('--frangi_cache', default=None,
                         help='Duong dan file .npz de cache Frangi maps, tranh tinh lai '
                              'moi lan chay (Frangi khong hoc, khong doi qua epoch).')
    parser.add_argument('--results_csv', default='results_region_topology.csv')
    parser.add_argument('--early_stop_unsupervised', action='store_true')
    parser.add_argument('--early_stop_warmup', type=int, default=3)
    parser.add_argument('--early_stop_window', type=int, default=5)
    parser.add_argument('--early_stop_unstable_count', type=int, default=3)
    parser.add_argument('--early_stop_threshold', type=float, default=0.20)
    return parser.parse_args()


def build_model(config):
    m = archs.__dict__[config['arch']](config['num_classes'],
                                        config['input_channels'],
                                        config['deep_supervision'])
    return m.cuda()


@torch.no_grad()
def ema_update(teacher, student, keep_rate):
    t_params = OrderedDict(teacher.named_parameters())
    s_params = OrderedDict(student.named_parameters())
    for name, t_p in t_params.items():
        t_p.data.mul_(keep_rate).add_(s_params[name].data, alpha=1 - keep_rate)
    t_buffers = OrderedDict(teacher.named_buffers())
    s_buffers = OrderedDict(student.named_buffers())
    for name, t_b in t_buffers.items():
        t_b.data.copy_(s_buffers[name].data)


@torch.no_grad()
def entropy_map(prob, eps=1e-8):
    p = prob.clamp(eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


@torch.no_grad()
def predicted_positive_ratio(teacher_r, teacher_t, image_only_loader, merge_kwargs):
    teacher_r.eval()
    teacher_t.eval()
    total_ratio, n = 0.0, 0
    for input in image_only_loader:
        input = input.cuda()
        pred_r = torch.sigmoid(teacher_r(input))
        pred_t = torch.sigmoid(teacher_t(input))
        ent_r = entropy_map(pred_r)
        merged = selective_merge_pseudo_label(pred_r, pred_t, ent_r, **merge_kwargs)
        ratio = (merged > 0.5).float().mean().item()
        total_ratio += ratio * input.size(0)
        n += input.size(0)
    teacher_r.train()
    teacher_t.train()
    return total_ratio / n


class DualPairRollback:
    """Ban rut gon cua UnsupervisedEarlyStopper, mo rong cho 2 CAP model."""

    def __init__(self, warmup_epochs, window_size, unstable_count_threshold,
                 ratio_change_threshold):
        self.warmup_epochs = warmup_epochs
        self.window_size = window_size
        self.unstable_count_threshold = unstable_count_threshold
        self.ratio_change_threshold = ratio_change_threshold
        self.ratio_history = []
        self.baseline = None
        self.recent_stability = deque(maxlen=window_size)
        self.best_state = None
        self.best_epoch = -1

    def check(self, teacher_r, teacher_t, image_only_loader, merge_kwargs, epoch_idx):
        ratio = predicted_positive_ratio(teacher_r, teacher_t, image_only_loader, merge_kwargs)
        self.ratio_history.append(ratio)

        if len(self.ratio_history) <= self.warmup_epochs:
            self.best_state = {
                'teacher_r': copy.deepcopy(teacher_r.state_dict()),
                'teacher_t': copy.deepcopy(teacher_t.state_dict()),
            }
            self.best_epoch = epoch_idx
            if len(self.ratio_history) == self.warmup_epochs:
                sorted_hist = sorted(self.ratio_history)
                mid = len(sorted_hist) // 2
                self.baseline = sorted_hist[mid]
            print(f"  [DualPairRollback] epoch {epoch_idx}: ty le du doan(gop)={ratio:.4f} "
                  f"(warmup {len(self.ratio_history)}/{self.warmup_epochs})")
            return False

        change = abs(ratio - self.baseline) / max(self.baseline, 1e-6)
        is_unstable = change > self.ratio_change_threshold
        self.recent_stability.append(is_unstable)
        unstable_count = sum(self.recent_stability)

        status = "BAT ON" if is_unstable else "ON DINH"
        print(f"  [DualPairRollback] epoch {epoch_idx}: ty le du doan(gop)={ratio:.4f}, "
              f"baseline={self.baseline:.4f}, lech={change*100:.1f}% -> {status} "
              f"(bat on {unstable_count}/{len(self.recent_stability)} trong "
              f"{self.window_size} epoch gan nhat)")

        if not is_unstable:
            self.best_state = {
                'teacher_r': copy.deepcopy(teacher_r.state_dict()),
                'teacher_t': copy.deepcopy(teacher_t.state_dict()),
            }
            self.best_epoch = epoch_idx

        if unstable_count >= self.unstable_count_threshold:
            print(f"  [DualPairRollback] DUNG SOM tai epoch {epoch_idx}. "
                  f"Quay ve checkpoint epoch {self.best_epoch}.")
            return True
        return False

    def restore_best(self, teacher_r, teacher_t):
        if self.best_state is not None:
            teacher_r.load_state_dict(self.best_state['teacher_r'])
            teacher_t.load_state_dict(self.best_state['teacher_t'])
        return self.best_epoch


def train_epoch(train_loader, student_r, teacher_r, opt_r,
                 student_t, teacher_t, opt_t,
                 criterion_seg, cldice_loss, lambda_cl, ema_keep_rate, merge_kwargs,
                 frangi_maps=None, frangi_threshold=FRANGI_BINARY_THRESHOLD_DEFAULT):
    avg = {'loss_r': AverageMeter(), 'loss_t': AverageMeter()}
    student_r.train()
    student_t.train()
    teacher_r.eval()
    teacher_t.eval()
    pbar = tqdm(total=len(train_loader))

    for batch in train_loader:
        input, _, img_id = batch
        w_input = input.cuda()
        image_strong_aug = build_strong_augmentation(input.squeeze(0))
        s_input = image_strong_aug.unsqueeze(0).cuda()

        with torch.no_grad():
            pred_r_logits, msrc_feat_r = teacher_r(w_input, mode='const')
            pred_t_logits, _ = teacher_t(w_input, mode='const')
            pred_r = torch.sigmoid(pred_r_logits)
            pred_t = torch.sigmoid(pred_t_logits)
            ent_r = entropy_map(pred_r)

            if frangi_maps is not None:
                # Dataset tra ve meta dang {'img_id': img_id}; qua DataLoader
                # (batch_size=1, default_collate) thanh {'img_id': [id_string]}.
                iid = img_id['img_id'][0] if isinstance(img_id, dict) else img_id
                fmap_np = frangi_maps.get(iid)
                if fmap_np is not None:
                    fmap = torch.from_numpy(fmap_np).float().unsqueeze(0).unsqueeze(0).cuda()
                    merged = frangi_assisted_merge(
                        pred_r, pred_t, ent_r, fmap,
                        lambda_thresh=(merge_kwargs['lambda_thresh'][0], merge_kwargs['lambda_thresh'][1]),
                        frangi_confidence_thresh=frangi_threshold)
                else:
                    merged = selective_merge_pseudo_label(pred_r, pred_t, ent_r, **merge_kwargs)
            else:
                merged = selective_merge_pseudo_label(pred_r, pred_t, ent_r, **merge_kwargs)

            ps_label = merged.clone()
            ps_label[ps_label >= 0.5] = 1
            ps_label[ps_label < 0.5] = 0

        opt_r.zero_grad()
        out_r, feat_r = student_r(s_input, mode='const')
        seg_loss_r = criterion_seg(out_r, ps_label)
        loss_r = seg_loss_r + consistency_loss(msrc_feat_r, feat_r)
        loss_r.backward()
        opt_r.step()

        opt_t.zero_grad()
        out_t, feat_t = student_t(s_input, mode='const')
        seg_loss_t = criterion_seg(out_t, ps_label)
        if lambda_cl > 0:
            cl = cldice_loss(out_t, ps_label)
            seg_loss_t = seg_loss_t + lambda_cl * cl
        loss_t = seg_loss_t + consistency_loss(msrc_feat_r, feat_t)
        loss_t.backward()
        opt_t.step()

        ema_update(teacher_r, student_r, ema_keep_rate)
        ema_update(teacher_t, student_t, ema_keep_rate)

        avg['loss_r'].update(loss_r.item(), input.size(0))
        avg['loss_t'].update(loss_t.item(), input.size(0))
        pbar.set_postfix(OrderedDict([('loss_r', avg['loss_r'].avg), ('loss_t', avg['loss_t'].avg)]))
        pbar.update(1)
    pbar.close()
    return {'loss_r': avg['loss_r'].avg, 'loss_t': avg['loss_t'].avg}


@torch.no_grad()
def evaluate_merged(val_loader, teacher_r, teacher_t, merge_kwargs):
    teacher_r.eval()
    teacher_t.eval()
    avg_iou, avg_dice, n = 0.0, 0.0, 0
    for input, target, _ in val_loader:
        input = input.cuda()
        target = target.cuda()
        pred_r_logits, _ = teacher_r(input, mode='const')
        pred_t_logits, _ = teacher_t(input, mode='const')
        pred_r = torch.sigmoid(pred_r_logits)
        pred_t = torch.sigmoid(pred_t_logits)
        ent_r = entropy_map(pred_r)
        merged = selective_merge_pseudo_label(pred_r, pred_t, ent_r, **merge_kwargs)
        iou, dice = iou_score(merged, target)
        avg_iou += iou * input.size(0)
        avg_dice += dice * input.size(0)
        n += input.size(0)
    return avg_iou / n, avg_dice / n


def main():
    args = parse_args()
    set_seed(args.seed)

    config_file = "config_" + args.target + "_dualema"
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_img_ids = glob(os.path.join('inputs', args.target, 'train', 'images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
    val_img_ids = glob(os.path.join('inputs', args.target, 'test', 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    train_transform = Compose([
        RandomRotate90(), transforms.Flip(),
        Resize(config['input_h'], config['input_w']), transforms.Normalize(),
    ])
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']), transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids, img_dir=os.path.join('inputs', args.target, 'train', 'images'),
        mask_dir=os.path.join('inputs', args.target, 'train', 'masks'),
        img_ext=config['img_ext'], mask_ext=config['mask_ext'],
        num_classes=config['num_classes'], transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=config['num_workers'], drop_last=True)

    val_dataset = Dataset(
        img_ids=val_img_ids, img_dir=os.path.join('inputs', args.target, 'test', 'images'),
        mask_dir=os.path.join('inputs', args.target, 'test', 'masks'),
        img_ext=config['img_ext'], mask_ext=config['mask_ext'],
        num_classes=config['num_classes'], transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'], drop_last=False)

    print("Loading source trained model...!!!")
    msrc_model = build_model(config)
    msrc_model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    msrc_model.eval()

    criterion = losses.__dict__[config['loss']]().cuda()
    print("\nPerforming source only model evaluation...!!!")
    val_log = validate(val_loader, msrc_model, criterion)
    source_only_dice = val_log['dice']
    print('Source_only dice: %.4f' % source_only_dice)

    if args.stage1_ckpt is not None and os.path.exists(args.stage1_ckpt):
        print(f"[CACHE] Tai checkpoint Stage I co san tu {args.stage1_ckpt}")
        msrc_model.load_state_dict(torch.load(args.stage1_ckpt))
    else:
        pseudo_model = build_model(config)
        pseudo_model.load_state_dict(msrc_model.state_dict())
        pseudo_model.eval()
        msrc_model.train()
        src_optimizer = optim.Adam(msrc_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        print("\nTarget specific adaptation (Stage I)...!!!")
        for epoch in range(config['stage1']):
            train_log = sfuda_target(config, train_loader, pseudo_model, msrc_model, criterion, src_optimizer)
            print('train_loss %.4f - train_iou %.4f' % (train_log['loss'], train_log['iou']))
        if args.stage1_ckpt is not None:
            os.makedirs(os.path.dirname(args.stage1_ckpt) or '.', exist_ok=True)
            torch.save(msrc_model.state_dict(), args.stage1_ckpt)
    msrc_model.eval()

    student_r = build_model(config)
    student_r.load_state_dict(msrc_model.state_dict())
    teacher_r = build_model(config)
    teacher_r.load_state_dict(msrc_model.state_dict())

    student_t = build_model(config)
    student_t.load_state_dict(msrc_model.state_dict())
    teacher_t = build_model(config)
    teacher_t.load_state_dict(msrc_model.state_dict())

    opt_r = optim.Adam(student_r.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    opt_t = optim.Adam(student_t.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    cldice_loss = SoftClDiceLoss(num_iter=args.cldice_num_iter)
    merge_kwargs = dict(lambda_thresh=(args.merge_lambda1, args.merge_lambda2))

    n_epochs = args.stage2_epochs if args.stage2_epochs is not None else config['stage2']
    print(f"\n[INFO] Stage II (Region/Topology Teacher) se chay {n_epochs} epoch")

    frangi_maps = None
    if args.use_frangi:
        cache_path = args.frangi_cache or f'cache/frangi_{args.target}.npz'
        frangi_maps = precompute_frangi_maps(
            img_dir=os.path.join('inputs', args.target, 'train', 'images'),
            img_ids=train_img_ids, img_ext=config['img_ext'],
            input_h=config['input_h'], input_w=config['input_w'],
            cache_path=cache_path)
        print(f"[FRANGI] Da bat Frangi-assisted merge, threshold={args.frangi_threshold}")

    rollback = DualPairRollback(
        warmup_epochs=args.early_stop_warmup, window_size=args.early_stop_window,
        unstable_count_threshold=args.early_stop_unstable_count,
        ratio_change_threshold=args.early_stop_threshold) if args.early_stop_unsupervised else None

    image_only_loader = None
    if rollback is not None:
        from unsupervised_early_stop import ImageOnlyDataset
        image_only_dataset = ImageOnlyDataset(
            img_ids=train_img_ids, img_dir=os.path.join('inputs', args.target, 'train', 'images'),
            mask_dir=os.path.join('inputs', args.target, 'train', 'masks'),
            img_ext=config['img_ext'], mask_ext=config['mask_ext'],
            num_classes=config['num_classes'], transform=val_transform)
        image_only_loader = torch.utils.data.DataLoader(
            image_only_dataset, batch_size=4, shuffle=False, num_workers=config['num_workers'])

    actual_epochs_run = n_epochs
    for epoch in range(n_epochs):
        train_log = train_epoch(train_loader, student_r, teacher_r, opt_r,
                                 student_t, teacher_t, opt_t,
                                 criterion, cldice_loss, args.lambda_cl,
                                 args.ema_keep_rate, merge_kwargs,
                                 frangi_maps=frangi_maps,
                                 frangi_threshold=args.frangi_threshold)
        print('epoch %d/%d - loss_r %.4f - loss_t %.4f' %
              (epoch + 1, n_epochs, train_log['loss_r'], train_log['loss_t']))

        if rollback is not None:
            should_stop = rollback.check(teacher_r, teacher_t, image_only_loader, merge_kwargs, epoch + 1)
            if should_stop:
                restored = rollback.restore_best(teacher_r, teacher_t)
                actual_epochs_run = restored
                break

    print("\nPerforming adapted target model evaluation (merged)...!!!")
    _, merged_dice = evaluate_merged(val_loader, teacher_r, teacher_t, merge_kwargs)
    _, dice_r_only = evaluate_merged(val_loader, teacher_r, teacher_r, merge_kwargs)
    print('Merged (Teacher_R + Teacher_T) dice: %.4f' % merged_dice)
    print('Teacher_R only dice (xap xi): %.4f' % dice_r_only)

    # --- DEBUG TAM: kiem tra Teacher_R va Teacher_T co thuc su khac nhau khong ---
    with torch.no_grad():
        sample_input, _, _ = next(iter(val_loader))
        sample_input = sample_input.cuda()
        out_r, _ = teacher_r(sample_input, mode='const')
        out_t, _ = teacher_t(sample_input, mode='const')
        diff = (torch.sigmoid(out_r) - torch.sigmoid(out_t)).abs()
        print(f"[DEBUG] Chenh lech trung binh |Teacher_R - Teacher_T| = {diff.mean().item():.6f}")
        print(f"[DEBUG] Chenh lech toi da = {diff.max().item():.6f}")
        pred_r_bin = (torch.sigmoid(out_r) > 0.5).float()
        pred_t_bin = (torch.sigmoid(out_t) > 0.5).float()
        bin_diff = (pred_r_bin - pred_t_bin).abs().mean().item()
        print(f"[DEBUG] Ty le pixel KHAC NHAU sau khi nhi phan hoa (>0.5): {bin_diff:.6f}")
    # --- HET DEBUG ---

    os.makedirs(os.path.dirname(args.results_csv) or '.', exist_ok=True)
    write_header = not os.path.exists(args.results_csv)
    with open(args.results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'source', 'target', 'seed', 'lambda_cl',
                              'ema_keep_rate', 'source_only_dice', 'merged_dice',
                              'teacher_r_only_dice', 'stage2_epochs_planned', 'stage2_epochs_actual'])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.source, args.target,
                          args.seed, args.lambda_cl, args.ema_keep_rate, source_only_dice,
                          merged_dice, dice_r_only, n_epochs, actual_epochs_run])
    print('Da ghi ket qua vao', args.results_csv)


if __name__ == '__main__':
    main()
