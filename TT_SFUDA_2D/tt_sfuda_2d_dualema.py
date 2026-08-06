"""
tt_sfuda_2d_dualema.py (v2)
============================
Them 2 tham so CLI so voi ban v1, de sweep nhanh khong can sua file:
  --ensemble_mode {mean, weighted, confidence}  (chi anh huong topology=parallel)
  --slow_keep_rate <float>   (ghi de keep_rate cua teacher 'slow' trong yaml,
                               vd de sweep 0.993/0.995/0.997/0.999)
  --fast_weight <float>      (dung khi ensemble_mode='weighted', trong so cua
                               teacher 'fast'; trong so 'slow' = 1 - fast_weight)

Cach chay (vi du sweep nhanh keep_rate cho cascaded/parallel):
    python run.py tt_sfuda_2d_dualema.py --source chase_unet --target rite \
        --topology parallel --ensemble_mode confidence

    python run.py tt_sfuda_2d_dualema.py --source chase_unet --target rite \
        --topology parallel --ensemble_mode weighted --fast_weight 0.8

    python run.py tt_sfuda_2d_dualema.py --source chase_unet --target rite \
        --topology cascaded --slow_keep_rate 0.995
"""
import os
import json
import yaml
import argparse
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

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

from tt_sfuda_2d import (
    build_strong_augmentation,
    consistency_loss,
    sfuda_target,
    validate,
)

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--topology', default='parallel', choices=['parallel', 'cascaded'])
    parser.add_argument('--ensemble_mode', default='mean', choices=['mean', 'weighted', 'confidence'])
    parser.add_argument('--slow_keep_rate', type=float, default=None,
                         help='Neu dat, ghi de keep_rate cua teacher "slow" tu yaml')
    parser.add_argument('--fast_weight', type=float, default=None,
                         help='Neu dat, ghi de weight cua teacher "fast" (dung voi ensemble_mode=weighted)')
    parser.add_argument('--results_csv', default='results_dualema.csv',
                         help='File CSV de append ket qua, phuc vu sweep nhieu lan')
    return parser.parse_args()


def sfuda_task_multiteacher(train_loader, teacher_manager, tgt_model, criterion, optimizer, ensemble_mode):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    teacher_manager.train_mode(False)
    tgt_model.train()
    pbar = tqdm(total=len(train_loader))

    for input, target, _ in train_loader:
        w_input = input.cuda()
        target = target.cuda()
        image_strong_aug = build_strong_augmentation(input.squeeze(0))
        s_input = image_strong_aug.unsqueeze(0).cuda()

        with torch.no_grad():
            w_output, msrc_feat = teacher_manager.predict(w_input, mode='const', ensemble_mode=ensemble_mode)
            ps_output = w_output.detach().clone()
            ps_output[ps_output >= 0.5] = 1
            ps_output[ps_output < 0.5] = 0

        optimizer.zero_grad()
        output, tgt_feat = tgt_model(s_input, mode='const')
        seg_loss = criterion(output, ps_output)
        const_loss = consistency_loss(msrc_feat, tgt_feat)
        loss = seg_loss + const_loss
        loss.backward()
        optimizer.step()

        iou, dice = iou_score(output, target)
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

        teacher_manager.update(tgt_model)

    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                         ('iou', avg_meters['iou'].avg)])


def main():
    args = parse_args()

    config_file = "config_" + args.target + "_dualema"
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # ---- ghi de teacher config qua CLI, khong can sua yaml moi lan sweep ----
    teachers = config['teachers']
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

    pseudo_model = archs.__dict__[config['arch']](config['num_classes'],
                                                   config['input_channels'],
                                                   config['deep_supervision'])
    pseudo_model.load_state_dict(msrc_model.state_dict())
    pseudo_model.cuda()
    pseudo_model.eval()

    criterion = losses.__dict__[config['loss']]().cuda()

    print("")
    print("Performing source only model evaluation...!!!")
    val_log = validate(val_loader, msrc_model, criterion)
    source_only_dice = val_log['dice']   # luu rieng - val_log se bi ghi de sau khi adapt xong
    print('Source_only dice: %.4f' % (source_only_dice))

    print("")
    print("Target specific adaptation (Stage I - giong het baseline)...!!!")
    for epoch in range(config['stage1']):
        train_log = sfuda_target(config, train_loader, pseudo_model, msrc_model, criterion, src_optimizer)
        print('train_loss %.4f - train_iou %.4f' % (train_log['loss'], train_log['iou']))

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
    for epoch in range(config['stage2']):
        train_log = sfuda_task_multiteacher(train_loader, teacher_manager, tgt_model, criterion,
                                             tgt_optimizer, args.ensemble_mode)
        print('train_loss %.4f - train_iou %.4f' % (train_log['loss'], train_log['iou']))

    print("")
    print("Performing adapted target model evaluation...!!!")
    val_log = validate(val_loader, tgt_model, criterion)
    print('Adapted target model (multi-teacher, %s, %s) dice: %.4f' %
          (args.topology, args.ensemble_mode, val_log['dice']))

    tag = f"{args.topology}_{args.ensemble_mode}"
    if args.slow_keep_rate is not None:
        tag += f"_slowkr{args.slow_keep_rate}"
    if args.fast_weight is not None:
        tag += f"_fw{args.fast_weight}"

    out_dir = f"outputs/{config['name']}_dualema_{tag}_to_{args.target}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(tgt_model.state_dict(), os.path.join(out_dir, 'model.pth'))

    # ---- ghi ket qua vao CSV de tien tong hop, khong ghi de - append ----
    row = {
        'source': args.source, 'target': args.target,
        'topology': args.topology, 'ensemble_mode': args.ensemble_mode,
        'slow_keep_rate': args.slow_keep_rate, 'fast_weight': args.fast_weight,
        'source_only_dice': source_only_dice,
        'adapted_dice': val_log['dice'],
    }
    write_header = not os.path.exists(args.results_csv)
    with open(args.results_csv, 'a') as f:
        if write_header:
            f.write(','.join(row.keys()) + '\n')
        f.write(','.join(str(v) for v in row.values()) + '\n')
    print(f"Da ghi ket qua vao {args.results_csv}")


if __name__ == '__main__':
    main()