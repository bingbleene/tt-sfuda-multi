"""
tt_sfuda_2d_dualema.py (v3)
=============================
So voi v2, them 2 co che quan trong de KET QUA CO THE SO SANH DUOC giua
cac lan chay (khong con bi nhieu RNG lam sai lech ket luan):

  1. --seed <int> (mac dinh 42) - co dinh torch/numpy/random seed TRUOC
     Stage I va Stage II, giam nhieu tu augmentation/shuffle.

  2. --stage1_ckpt <path> - NEU file da ton tai, BO QUA train lai Stage I,
     load thang checkpoint co san va di thang vao Stage II. NEU chua ton
     tai, train Stage I nhu binh thuong ROI LUU LAI vao duong dan nay, de
     lan sau (voi topology/keep_rate khac) tai su dung - dam bao MOI THU
     NGHIEM STAGE II XUAT PHAT TU CUNG 1 DIEM, chi khac nhau o phan dang
     thuc su muon so sanh (Stage II).

Cach dung (khuyen nghi TU GIO VE SAU thay vi goi truc tiep nhu v2):
    # Lan dau tien cho 1 domain shift - se train Stage I va luu lai
    python run.py tt_sfuda_2d_dualema.py --source chase_unet --target rite \
        --topology parallel --slow_keep_rate 0.995 \
        --stage1_ckpt cache/chase_to_rite_stage1.pth

    # Cac lan sau, CUNG domain shift, khac topology/keep_rate - TAI SU DUNG
    # checkpoint Stage I, khong train lai, loai bo nhieu:
    python run.py tt_sfuda_2d_dualema.py --source chase_unet --target rite \
        --topology cascaded --stage1_ckpt cache/chase_to_rite_stage1.pth
"""
import os
import random
import yaml
import argparse
from datetime import datetime
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

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
from multi_teacher import MultiTeacherManager
from masked_loss import MaskedBCEDiceLoss
from class_balance import ClassBalanceTracker, CalibratedBCEDiceLoss
from unsupervised_early_stop import UnsupervisedEarlyStopper, ImageOnlyDataset

from tt_sfuda_2d import (
    build_strong_augmentation,
    consistency_loss,
    sfuda_target,
    validate,
)

cudnn.benchmark = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # luu y: cudnn.benchmark=True (o tren) co the van gay nhieu do chon
    # thuat toan nhanh nhat tuy phan cung - chap nhan duoc, day la nguon
    # nhieu con lai nho hon nhieu so voi RNG cua augmentation/shuffle.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--topology', default='parallel', choices=['parallel', 'cascaded'])
    parser.add_argument('--ensemble_mode', default='mean',
                         choices=['mean', 'weighted', 'confidence', 'warmup', 'trust_region'])
    parser.add_argument('--slow_keep_rate', type=float, default=None)
    parser.add_argument('--fast_weight', type=float, default=None)
    parser.add_argument('--results_csv', default='results_dualema.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage1_ckpt', default=None,
                         help='Neu dat: cache/tai su dung checkpoint Stage I, '
                              'dam bao moi thu nghiem Stage II xuat phat cung 1 diem.')
    parser.add_argument('--single_teacher', action='store_true',
                         help='Neu dat: CHI dung teacher "fast" (keep_rate=0.99), '
                              'tuong duong toan hoc voi baseline 1-teacher goc - '
                              'dung de so sanh cong bang qua CUNG 1 stage1_ckpt.')
    parser.add_argument('--class_balance', action='store_true',
                         help='Neu dat: bat class-balance loss calibration (xem class_balance.py).')
    parser.add_argument('--class_balance_strategy', default='symmetric',
                         choices=['cbmt', 'symmetric', 'fixed_fg'],
                         help='cbmt=cong thuc goc CBMT (chi giam bg); '
                              'symmetric=ben nao kho hon duoc tang trong so; '
                              'fixed_fg=tang co dinh trong so tien canh, khong can theo doi dong')
    parser.add_argument('--fixed_fg_weight', type=float, default=2.0,
                         help='Chi dung khi class_balance_strategy=fixed_fg')
    parser.add_argument('--stage2_epochs', type=int, default=None,
                         help='Neu dat: GHI DE so epoch Stage II tu config goc '
                              '(vd config goc chi co 5, thu keo dai len 10-15-20 '
                              'xem Dual-EMA + class-balance co du on dinh de train '
                              'lau hon khong bi suy thoai hay khong)')
    parser.add_argument('--early_stop_unsupervised', action='store_true',
                         help='Bat early-stopping KHONG GIAM SAT (khong dung label '
                              'dich) - dua tren do troi ty le pixel du doan duong '
                              'tinh. Dung KET HOP voi --stage2_epochs de dat SO '
                              'EPOCH LON, roi de co che nay tu dung dung som neu '
                              'phat hien bat on.')
    parser.add_argument('--early_stop_warmup', type=int, default=3,
                         help='So epoch dau dung tinh baseline CO DINH (khong doi sau do)')
    parser.add_argument('--early_stop_window', type=int, default=5,
                         help='Kich thuoc cua so truot de dem epoch bat on GAN DAY')
    parser.add_argument('--early_stop_unstable_count', type=int, default=3,
                         help='So epoch bat on TRONG cua so gan nhat de kich hoat dung (khong can lien tiep)')
    parser.add_argument('--early_stop_threshold', type=float, default=0.20)
    return parser.parse_args()


def sfuda_task_multiteacher(train_loader, teacher_manager, tgt_model, criterion, optimizer,
                             ensemble_mode, class_balance_tracker=None, labels_available=True):
    """
    labels_available=False: train_loader la loader ANH THUAN TUY (khong co
    mask, vd tu ImageOnlyDataset) - dung khi thu nghiem early_stop_unsupervised,
    de dam bao KHONG co label nao bi cham vao trong SUOT vong lap Stage II,
    ke ca chi de log train_iou (von von KHONG dung trong loss/backward, nhung
    van "cham" vao label neu con doc no tu dia).
    """
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    teacher_manager.train_mode(False)
    tgt_model.train()
    pbar = tqdm(total=len(train_loader))
    masked_criterion = MaskedBCEDiceLoss() if ensemble_mode == 'trust_region' else None
    calibrated_criterion = CalibratedBCEDiceLoss() if class_balance_tracker is not None else None
    mean_trust_ratio = AverageMeter()

    for batch in train_loader:
        if labels_available:
            input, target, _ = batch
            target = target.cuda()
        else:
            input = batch   # ImageOnlyDataset chi tra ve 1 tensor anh, khong co gi khac
            target = None

        w_input = input.cuda()
        image_strong_aug = build_strong_augmentation(input.squeeze(0))
        s_input = image_strong_aug.unsqueeze(0).cuda()

        with torch.no_grad():
            if ensemble_mode == 'trust_region':
                ps_output, trust_mask, msrc_feat = teacher_manager.predict_with_trust(w_input, mode='const')
                mean_trust_ratio.update(trust_mask.mean().item(), input.size(0))
            else:
                w_output, msrc_feat = teacher_manager.predict(w_input, mode='const', ensemble_mode=ensemble_mode)
                ps_output = w_output.detach().clone()
                ps_output[ps_output >= 0.5] = 1
                ps_output[ps_output < 0.5] = 0

        optimizer.zero_grad()
        output, tgt_feat = tgt_model(s_input, mode='const')

        if ensemble_mode == 'trust_region':
            seg_loss = masked_criterion(output, ps_output, trust_mask)
        elif class_balance_tracker is not None:
            fg_w, bg_w = class_balance_tracker.get_weights()
            seg_loss = calibrated_criterion(output, ps_output, fg_weight=fg_w, bg_weight=bg_w)
            class_balance_tracker.update(output.detach(), ps_output)
        else:
            seg_loss = criterion(output, ps_output)

        const_loss = consistency_loss(msrc_feat, tgt_feat)
        loss = seg_loss + const_loss
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        if labels_available:
            iou, dice = iou_score(output, target)
            avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
        if labels_available:
            postfix['iou'] = avg_meters['iou'].avg
        if ensemble_mode == 'trust_region':
            postfix['trust%'] = mean_trust_ratio.avg
        if class_balance_tracker is not None:
            fg_w, bg_w = class_balance_tracker.get_weights()
            postfix['fg_w'] = fg_w
            postfix['bg_w'] = bg_w
        pbar.set_postfix(postfix)
        pbar.update(1)

        teacher_manager.update(tgt_model)

    pbar.close()
    result = OrderedDict([('loss', avg_meters['loss'].avg)])
    if labels_available:
        result['iou'] = avg_meters['iou'].avg
    if ensemble_mode == 'trust_region':
        result['trust_ratio'] = mean_trust_ratio.avg
    if class_balance_tracker is not None:
        fg_w, bg_w = class_balance_tracker.get_weights()
        result['fg_weight'] = fg_w
        result['bg_weight'] = bg_w
    return result


def main():
    run_start_time = datetime.now()
    args = parse_args()
    set_seed(args.seed)

    config_file = "config_" + args.target + "_dualema"
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    teachers = config['teachers']
    if args.single_teacher:
        teachers = [t for t in teachers if t['name'] == 'fast']
        assert len(teachers) == 1, "Khong tim thay teacher 'fast' trong config"
    if args.slow_keep_rate is not None:
        for t in teachers:
            if t['name'] == 'slow':
                t['keep_rate'] = args.slow_keep_rate
    if args.fast_weight is not None:
        for t in teachers:
            if t['name'] == 'fast':
                t['weight'] = args.fast_weight
            elif t['name'] == 'slow':
                t['weight'] = 1.0 - args.fast_weight

    train_img_ids = glob(os.path.join('inputs', args.target, 'train', 'images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
    val_img_ids = glob(os.path.join('inputs', args.target, 'test', 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', args.target, 'train', 'images'),
        mask_dir=os.path.join('inputs', args.target, 'train', 'masks'),
        img_ext=config['img_ext'], mask_ext=config['mask_ext'],
        num_classes=config['num_classes'], transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=config['num_workers'], drop_last=True)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', args.target, 'test', 'images'),
        mask_dir=os.path.join('inputs', args.target, 'test', 'masks'),
        img_ext=config['img_ext'], mask_ext=config['mask_ext'],
        num_classes=config['num_classes'], transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=config['num_workers'], drop_last=False)

    print("Loading source trained model...!!!")
    msrc_model = archs.__dict__[config['arch']](config['num_classes'],
                                                 config['input_channels'],
                                                 config['deep_supervision'])
    msrc_model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    msrc_model.cuda()
    msrc_model.train()

    tgt_model = archs.__dict__[config['arch']](config['num_classes'],
                                                config['input_channels'],
                                                config['deep_supervision'])
    tgt_model.cuda()
    tgt_model.train()

    src_params = filter(lambda p: p.requires_grad, msrc_model.parameters())
    src_optimizer = optim.Adam(src_params, lr=config['lr'], weight_decay=config['weight_decay'])
    tgt_params = filter(lambda p: p.requires_grad, tgt_model.parameters())
    tgt_optimizer = optim.Adam(tgt_params, lr=config['lr'], weight_decay=config['weight_decay'])

    criterion = losses.__dict__[config['loss']]().cuda()

    print("")
    print("Performing source only model evaluation...!!!")
    val_log = validate(val_loader, msrc_model, criterion)
    source_only_dice = val_log['dice']
    print('Source_only dice: %.4f' % (source_only_dice))

    # ================== DIEM KHAC BIET CHINH so voi v2 ==================
    if args.stage1_ckpt is not None and os.path.exists(args.stage1_ckpt):
        print("")
        print(f"[CACHE] Tai checkpoint Stage I co san tu {args.stage1_ckpt} - BO QUA train lai.")
        msrc_model.load_state_dict(torch.load(args.stage1_ckpt))
    else:
        pseudo_model = archs.__dict__[config['arch']](config['num_classes'],
                                                       config['input_channels'],
                                                       config['deep_supervision'])
        pseudo_model.load_state_dict(msrc_model.state_dict())
        pseudo_model.cuda()
        pseudo_model.eval()

        print("")
        print("Target specific adaptation (Stage I)...!!!")
        for epoch in range(config['stage1']):
            train_log = sfuda_target(config, train_loader, pseudo_model, msrc_model, criterion, src_optimizer)
            print('train_loss %.4f - train_iou %.4f' % (train_log['loss'], train_log['iou']))

        if args.stage1_ckpt is not None:
            os.makedirs(os.path.dirname(args.stage1_ckpt) or '.', exist_ok=True)
            torch.save(msrc_model.state_dict(), args.stage1_ckpt)
            print(f"[CACHE] Da luu checkpoint Stage I vao {args.stage1_ckpt} de tai su dung sau nay.")
    # ======================================================================

    msrc_model.eval()
    tgt_model.load_state_dict(msrc_model.state_dict())
    tgt_model.cuda()
    tgt_model.train()

    print("")
    print(f"Khoi tao {len(teachers)} teacher, topology='{args.topology}', "
          f"ensemble_mode='{args.ensemble_mode}': {teachers}")
    teacher_manager = MultiTeacherManager(teachers, topology=args.topology)
    teacher_manager.init_from(msrc_model)
    teacher_manager.cuda()

    print("")
    print("Task specific adaptation (Stage II - Multi-EMA Teacher)...!!!")
    class_balance_tracker = ClassBalanceTracker(
        strategy=args.class_balance_strategy,
        fixed_fg_weight=args.fixed_fg_weight) if args.class_balance else None
    n_epochs = args.stage2_epochs if args.stage2_epochs is not None else config['stage2']
    print(f"[INFO] Stage II se chay {n_epochs} epoch "
          f"({'ghi de tu --stage2_epochs' if args.stage2_epochs is not None else 'tu config goc'})")

    early_stopper = UnsupervisedEarlyStopper(
        warmup_epochs=args.early_stop_warmup,
        window_size=args.early_stop_window,
        unstable_count_threshold=args.early_stop_unstable_count,
        ratio_change_threshold=args.early_stop_threshold) if args.early_stop_unsupervised else None

    image_only_loader = None
    if early_stopper is not None:
        # ImageOnlyDataset KE THUA Dataset goc (xem unsupervised_early_stop.py)
        # - dam bao pixel giong tuyet doi, chi khac o cho KHONG tra ve mask.
        # Van phai truyen mask_dir hop le (lop cha can de doc, du bi bo o
        # __getitem__ cua lop con) - day la du lieu THAT, khong phai gia.

        # image_only_loader: dung val_transform (KHONG augmentation ngau
        # nhien) - do ty le du doan can ON DINH qua cac epoch de phat hien
        # troi dat that su, tranh nhieu tu augmentation ngau nhien.
        image_only_ds = ImageOnlyDataset(
            img_ids=train_img_ids,
            img_dir=os.path.join('inputs', args.target, 'train', 'images'),
            mask_dir=os.path.join('inputs', args.target, 'train', 'masks'),
            img_ext=config['img_ext'], mask_ext=config['mask_ext'],
            num_classes=config['num_classes'], transform=val_transform)
        image_only_loader = torch.utils.data.DataLoader(image_only_ds, batch_size=4, shuffle=False, num_workers=2)

        # stage2_train_loader: dung train_transform (CO augmentation, giong
        # het du lieu train_loader goc dua vao model) - chi khac o cho
        # KHONG BAO GIO tra ve mask cho vong lap Stage II.
        stage2_train_ds = ImageOnlyDataset(
            img_ids=train_img_ids,
            img_dir=os.path.join('inputs', args.target, 'train', 'images'),
            mask_dir=os.path.join('inputs', args.target, 'train', 'masks'),
            img_ext=config['img_ext'], mask_ext=config['mask_ext'],
            num_classes=config['num_classes'], transform=train_transform)
        stage2_train_loader = torch.utils.data.DataLoader(
            stage2_train_ds, batch_size=1, shuffle=True, num_workers=config['num_workers'], drop_last=True)
    else:
        stage2_train_loader = train_loader

    actual_epochs_run = n_epochs

    for epoch in range(n_epochs):
        teacher_manager.set_progress(epoch / max(1, n_epochs - 1))
        train_log = sfuda_task_multiteacher(stage2_train_loader, teacher_manager, tgt_model, criterion,
                                             tgt_optimizer, args.ensemble_mode, class_balance_tracker,
                                             labels_available=(early_stopper is None))
        log_msg = 'epoch %d/%d - train_loss %.4f' % (epoch + 1, n_epochs, train_log['loss'])
        if 'iou' in train_log:
            log_msg += ' - train_iou %.4f' % train_log['iou']
        if 'fg_weight' in train_log:
            log_msg += ' - fg_w %.3f - bg_w %.3f' % (train_log['fg_weight'], train_log['bg_weight'])
        print(log_msg)

        # ==== THEO DOI DICE MOI EPOCH - CHI CHAY KHI KHONG bat early-stop ====
        # Neu dang thu nghiem early_stop_unsupervised, KHOA HAN buoc nay -
        # khong duoc phep cham vao val_loader (co label that) trong suot
        # vong lap Stage II, du chi de "tham khao". Chi khi KHONG bat co
        # che unsupervised (vd luc dieu tra hien tuong sup o stage2_epochs=20
        # ban dau) moi cho phep theo doi Dice moi epoch de chan doan.
        if not args.early_stop_unsupervised:
            tgt_model.eval()
            epoch_val_log = validate(val_loader, tgt_model, criterion)
            tgt_model.train()
            print('  -> [Theo doi - CHI THAM KHAO, khong dung de quyet dinh] Dice sau epoch %d: %.4f' %
                  (epoch + 1, epoch_val_log['dice']))

        if early_stopper is not None:
            should_stop = early_stopper.check(tgt_model, image_only_loader, epoch + 1)
            if should_stop:
                actual_epochs_run = epoch + 1
                restored_epoch = early_stopper.restore_best(tgt_model)
                print(f"[EarlyStop-KGS] DUNG SOM tai epoch {epoch+1} (phat hien bat on lien tiep). "
                      f"Quay ve checkpoint epoch {restored_epoch}.")
                break

    print("")
    print("Performing adapted target model evaluation...!!!")
    val_log = validate(val_loader, tgt_model, criterion)
    print('Adapted target model (multi-teacher, %s, %s) dice: %.4f' %
          (args.topology, args.ensemble_mode, val_log['dice']))

    duration_sec = (datetime.now() - run_start_time).total_seconds()

    tag = f"{args.topology}_{args.ensemble_mode}"
    if args.slow_keep_rate is not None:
        tag += f"_slowkr{args.slow_keep_rate}"
    if args.fast_weight is not None:
        tag += f"_fw{args.fast_weight}"

    out_dir = f"outputs/{config['name']}_dualema_{tag}_to_{args.target}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(tgt_model.state_dict(), os.path.join(out_dir, 'model.pth'))

    row = {
        'timestamp': run_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': args.source, 'target': args.target,
        'topology': args.topology, 'ensemble_mode': args.ensemble_mode,
        'slow_keep_rate': args.slow_keep_rate, 'fast_weight': args.fast_weight,
        'single_teacher': args.single_teacher,
        'class_balance': args.class_balance,
        'class_balance_strategy': args.class_balance_strategy if args.class_balance else None,
        'seed': args.seed, 'stage1_cached': args.stage1_ckpt is not None,
        'source_only_dice': source_only_dice,
        'adapted_dice': val_log['dice'],
        'duration_sec': round(duration_sec, 1),
        'stage2_epochs_planned': n_epochs,
        'stage2_epochs_actual': actual_epochs_run,
    }
    write_header = not os.path.exists(args.results_csv)
    with open(args.results_csv, 'a') as f:
        if write_header:
            f.write(','.join(row.keys()) + '\n')
        f.write(','.join(str(v) for v in row.values()) + '\n')
    print(f"Da ghi ket qua vao {args.results_csv}")


if __name__ == '__main__':
    main()
