"""
unsupervised_config_selector.py
==================================
GIAI QUYET LO HONG PHUONG PHAP QUAN TRONG NHAT: truoc day, resolution/CLAHE
duoc chon bang cach do Dice tren target ground-truth roi lay cau hinh tot
nhat - dieu nay VI PHAM gia dinh "target unlabeled" cua SFUDA (dung nhan
target de chon mo hinh). Script nay thay the bang 1 tieu chi HOAN TOAN
KHONG DUNG NHAN: entropy trung binh cua du doan SAU Stage I.

CO SO: sigmoid_entropy_loss() DA CO SAN trong tt_sfuda_2d.py, la thanh
phan chinh cua Stage I goc (paper gom ca EEM + selective voting). Day KHONG
PHAI phat minh moi - chi la TAI SU DUNG dung co che entropy minimization
paper da co, nhung dung no LAM TIEU CHI CHON CAU HINH thay vi chi lam
loss function.

NGUYEN LY: neu resolution/CLAHE giup model "nhin ro" domain dich hon (dung
nhu gia thuyet mat thong tin/khac mau sac), thi SAU Stage I, model se dua
ra du doan TU TIN HON (entropy thap hon) mot cach nhat quan - hoan toan
khong can biet du doan do co DUNG hay khong (khong can nhan).

CANH BAO QUAN TRONG - GIOI HAN CUA PHUONG PHAP NAY:
entropy thap KHONG dam bao Dice cao - model co the tu tin NHUNG SAI (chinh
la co che confirmation bias da thao luan). Vi vay entropy chi la PROXY, can
validate bang cach doi chieu (post-hoc, KHONG dung de chon) voi Dice that
da co san tu truoc, xem 2 thu co tuong quan khong. Neu tuong quan tot, day
la bang chung entropy la proxy hop le CHO CAC DOMAIN SHIFT TUONG TU trong
tuong lai (khi khong co san Dice de doi chieu).

Cach dung:
    python run.py unsupervised_config_selector.py --source chase_unet --target hrf \\
        --resolutions 512 768 1024 --clahe_options false true
"""
import os
import argparse
from glob import glob
from collections import OrderedDict

import cv2
import yaml
import numpy as np
import torch
import torch.optim as optim
from albumentations import RandomRotate90, Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

import archs
from dataset import Dataset
from clahe_transform import ClaheLAB
from tt_sfuda_2d import sfuda_target, sigmoid_entropy_loss, build_pseduo_augmentation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--resolutions', type=int, nargs='+', default=[512, 768, 1024])
    parser.add_argument('--clahe_options', type=str, nargs='+', default=['false', 'true'])
    parser.add_argument('--n_entropy_images', type=int, default=10,
                         help='So anh TRAIN dung de do entropy cuoi cung (khong can nhan, '
                              'nhung van dung tap train, khong dung tap test, giu tach biet).')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_model(config):
    m = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'],
                                        config['deep_supervision'])
    return m.cuda()


@torch.no_grad()
def measure_unsupervised_entropy(model, img_paths, input_h, input_w, use_clahe):
    """Do entropy trung binh cua du doan model tren 1 tap anh - HOAN TOAN
    KHONG dung nhan/mask o buoc nao. Day la con so DUY NHAT dung de xep
    hang cac cau hinh."""
    model.eval()
    normalize = transforms.Normalize()
    clahe = ClaheLAB(clip_limit=2.0) if use_clahe else None

    total_entropy, n = 0.0, 0
    for img_path in img_paths:
        img_bgr = cv2.imread(img_path)
        img_resized = cv2.resize(img_bgr, (input_w, input_h))
        if clahe is not None:
            img_resized = clahe(image=img_resized)['image']
        img_norm = normalize(image=img_resized)['image']
        img_norm = img_norm.astype('float32') / 255.0
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).cuda()

        output = model(tensor)
        ent = sigmoid_entropy_loss(torch.sigmoid(output))
        total_entropy += ent.item()
        n += 1

    return total_entropy / max(n, 1)


def run_one_config(source, target, resolution, use_clahe, n_entropy_images, seed):
    """Chay Stage I (dung DUNG sfuda_target da co san, khong viet lai) tai
    1 cau hinh (resolution, clahe), roi do entropy khong-nhan tren tap train."""
    torch.manual_seed(seed)

    config_file = f"config_{target}_dualema"
    with open(f'models/{source}/{config_file}.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_img_ids = glob(os.path.join('inputs', target, 'train', 'images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    transform_list = [RandomRotate90(), transforms.Flip()]
    if use_clahe:
        transform_list.append(ClaheLAB(clip_limit=2.0))
    transform_list += [Resize(resolution, resolution), transforms.Normalize()]
    train_transform = Compose(transform_list)

    train_dataset = Dataset(
        img_ids=train_img_ids, img_dir=os.path.join('inputs', target, 'train', 'images'),
        mask_dir=os.path.join('inputs', target, 'train', 'masks'),
        img_ext=config['img_ext'], mask_ext=config['mask_ext'],
        num_classes=config['num_classes'], transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=True)

    msrc_model = build_model(config)
    msrc_model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    pseudo_model = build_model(config)
    pseudo_model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))

    optimizer = optim.Adam(msrc_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = torch.nn.BCEWithLogitsLoss().cuda()

    # Stage I - DUNG NGUYEN sfuda_target() da co san, khong sua doi
    sfuda_target(config, train_loader, pseudo_model, msrc_model, criterion, optimizer)

    # Do entropy KHONG-NHAN tren tap anh rieng (khong lien quan buoc train o tren)
    entropy_img_paths = glob(os.path.join('inputs', target, 'train', 'images', '*' + config['img_ext']))[:n_entropy_images]
    entropy = measure_unsupervised_entropy(msrc_model, entropy_img_paths, resolution, resolution, use_clahe)

    return entropy


def main():
    args = parse_args()
    clahe_bools = [c.lower() == 'true' for c in args.clahe_options]

    results = []
    print(f"{'Resolution':<12} {'CLAHE':<8} {'Entropy khong-nhan (thap hon = tu tin hon)'}")
    print("-" * 65)
    for res in args.resolutions:
        for use_clahe in clahe_bools:
            entropy = run_one_config(args.source, args.target, res, use_clahe,
                                      args.n_entropy_images, args.seed)
            results.append((res, use_clahe, entropy))
            print(f"{res:<12} {str(use_clahe):<8} {entropy:.5f}")

    best = min(results, key=lambda x: x[2])
    print("\n" + "=" * 65)
    print(f"CAU HINH DUOC CHON (entropy thap nhat, KHONG dung nhan target): "
          f"resolution={best[0]}, clahe={best[1]} (entropy={best[2]:.5f})")
    print("=" * 65)
    print("\nLUU Y: day la lua chon THUAN entropy. Neu ban co san Dice that tu truoc,")
    print("chay them buoc doi chieu (khong dung de chon, chi de KIEM CHUNG do tin cay")
    print("cua tieu chi entropy nay) truoc khi tin tuong hoan toan.")


if __name__ == '__main__':
    main()
