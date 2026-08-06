"""
tt_sfuda_2d_dualema.py
=======================
Entrypoint cho Pha 5 (Dual/Multi-EMA Teacher). KHONG sua tt_sfuda_2d.py goc -
file nay import lai toan bo ham dung duoc (Stage I, augmentation, loss...)
tu tt_sfuda_2d.py, chi viet lai sfuda_task() de dung MultiTeacherManager
thay vi 1 teacher don.

Cach chay (dat cung thu muc voi archs.py, dataset.py, losses.py, utils.py,
multi_teacher.py va tt_sfuda_2d.py goc):

    python tt_sfuda_2d_dualema.py --source chase_unet --target rite \
        --topology parallel

    python tt_sfuda_2d_dualema.py --source chase_unet --target rite \
        --topology cascaded

Doc config tu models/<source>/config_<target>_dualema.yml (file MOI, xem
README_dualema.md de biet cac key can them: teachers, topology).
"""
import os
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

# tai su dung nguyen ven tu file goc - KHONG copy-paste lai logic
from tt_sfuda_2d import (
    build_strong_augmentation,
    build_pseduo_augmentation,
    consistency_loss,
    sigmoid_entropy_loss,
    uncert_voting,
    sfuda_target,   # Stage I giu nguyen, khong lien quan den teacher
    validate,
)

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--topology', default='parallel', choices=['parallel', 'cascaded'])
    return parser.parse_args()


def sfuda_task_multiteacher(train_loader, teacher_manager, tgt_model, criterion, optimizer):
    """
    Ban sao cua sfuda_task() goc (tt_sfuda_2d.py dong 184-226), CHI khac o
    2 diem:
      1. msrc_model (1 teacher) -> teacher_manager (N teacher, xem multi_teacher.py)
      2. update_teacher_model(...) don le -> teacher_manager.update(...)
    Phan con lai (augmentation, seg_loss, const_loss) giu NGUYEN logic goc.
    """
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    teacher_manager.train_mode(False)  # tat ca teacher o eval mode
    tgt_model.train()
    pbar = tqdm(total=len(train_loader))

    for input, target, _ in train_loader:
        w_input = input.cuda()
        target = target.cuda()
        image_strong_aug = build_strong_augmentation(input.squeeze(0))
        s_input = image_strong_aug.unsqueeze(0).cuda()

        with torch.no_grad():
            w_output, msrc_feat = teacher_manager.predict(w_input, mode='const')
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

        teacher_manager.update(tgt_model)   # <-- khac biet duy nhat so voi ban goc

    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                         ('iou', avg_meters['iou'].avg)])


def main():
    args = parse_args()

    config_file = "config_" + args.target + "_dualema"
    with open('models/%s/%s.yml' % (args.source, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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
    print('Source_only dice: %.4f' % (val_log['dice']))

    print("")
    print("Target specific adaptation (Stage I - giong het baseline)...!!!")
    for epoch in range(config['stage1']):
        train_log = sfuda_target(config, train_loader, pseudo_model, msrc_model, criterion, src_optimizer)
        print('train_loss %.4f - train_iou %.4f' % (train_log['loss'], train_log['iou']))

    msrc_model.eval()
    tgt_model.load_state_dict(msrc_model.state_dict())
    tgt_model.cuda()
    tgt_model.train()

    # ==== Diem khac biet chinh: khoi tao MultiTeacherManager thay vi 1 teacher ====
    print("")
    print(f"Khoi tao {len(config['teachers'])} teacher, topology='{args.topology}': "
          f"{[(t['name'], t['keep_rate']) for t in config['teachers']]}")
    teacher_manager = MultiTeacherManager(config['teachers'], topology=args.topology)
    teacher_manager.init_from(msrc_model)
    teacher_manager.cuda()

    print("")
    print("Task specific adaptation (Stage II - Multi-EMA Teacher)...!!!")
    for epoch in range(config['stage2']):
        train_log = sfuda_task_multiteacher(train_loader, teacher_manager, tgt_model, criterion, tgt_optimizer)
        print('train_loss %.4f - train_iou %.4f' % (train_log['loss'], train_log['iou']))

    print("")
    print("Performing adapted target model evaluation...!!!")
    val_log = validate(val_loader, tgt_model, criterion)
    print('Adapted target model (multi-teacher, %s) dice: %.4f' % (args.topology, val_log['dice']))

    out_dir = f"outputs/{config['name']}_dualema_{args.topology}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(tgt_model.state_dict(), os.path.join(out_dir, 'model.pth'))
    print(f"Da luu checkpoint tai {out_dir}/model.pth")


if __name__ == '__main__':
    main()