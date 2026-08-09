"""
cross_arch_bottleneck.py
==========================
Kien truc Teacher_X cho huong (a) Cross-architecture co-training.

VI SAO THIET KE THE NAY (khac Region/Topology Teacher cu - da that bai vi
Teacher_R va Teacher_T CHI khac loss, cung kien truc VGGBlock 5-level, nen
EMA rate 0.99 hoa tan het su khac biet truoc khi kip phan ky - chenh lech
trung binh do duoc chi ~0.002-0.003).

O day Teacher_X GIU NGUYEN encoder/decoder VGGBlock 5-level HET (conv0_0 ->
conv0_4, final) GIONG HET archs.UNet - de:
  1. Co the load truc tiep state_dict tu stage1 checkpoint (msrc_model) cho
     TAT CA cac layer nay, khong mat kien thuc da hoc duoc o Stage I.
  2. Giu nguyen shape 4 feature map [x1_0,x2_0,x3_0,x4_0] (64,128,256,512
     channel) => consistency_loss() va selective_merge_pseudo_label() hien
     co dung THANG, khong can sua gi.

Diem KHAC BIET THAT SU (khong bi EMA hoa tan) nam o buoc xu ly x4_0
(bottleneck, noi quyet dinh receptive field lon nhat cua toan mang):
them mot nhanh ASPP (Atrous Spatial Pyramid Pooling, dilation 1/2/4/8)
CONG DON (residual, khong thay the) vao x4_0. Day la thay doi KIEN TRUC,
khong phai loss - nen khong bi EMA (chi lam muot theo thoi gian tren CUNG
1 kien truc) xoa di.

QUAN TRONG: KHONG zero-init lop cuoi cua ASPP. Neu zero-init, luc epoch 0
Teacher_X se cho output gan y het Teacher_R (vi nhanh ASPP dong gop ~0),
va can nhieu epoch moi phan ky dan - dung y het van de cu (EMA/train qua
it buoc de kip phan ky tren data nho). Init ASPP binh thuong (Kaiming mac
dinh cua Conv2d) de co su khac biet hanh vi NGAY TU DAU.

CACH DUNG:
    from cross_arch_bottleneck import UNetDilatedBottleneck, load_from_unet_checkpoint

    student_x = UNetDilatedBottleneck(config['num_classes'], config['input_channels'],
                                       config['deep_supervision']).cuda()
    load_from_unet_checkpoint(student_x, 'models/chase_unet/model.pth')  # hoac stage1_ckpt
    teacher_x = UNetDilatedBottleneck(...).cuda()
    load_from_unet_checkpoint(teacher_x, 'models/chase_unet/model.pth')

Roi dung HET nhu Teacher_T cu trong tt_sfuda_2d_region_topology.py (cung
selective_merge_pseudo_label, cung consistency_loss, cung DualPairRollback -
khong can sua cac ham do vi interface (forward tra ve (logits, feat_list))
giong het archs.UNet).
"""

import torch
import torch.nn as nn


# Giu nguyen VGGBlock tu archs.py (copy lai vi import truc tiep tu archs se
# keo theo __all__ = ['UNet'] gay nham lan ten khi archs.__dict__ duoc dung
# de build model theo config['arch'] o noi khac trong pipeline).
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class ASPPResidual(nn.Module):
    """4 nhanh dilated conv (rate 1,2,4,8) + project ve lai so channel goc,
    CONG DON (residual) vao input - khong thay the, chi bo sung ngu canh
    da ty le. Giu nguyen channel count nen ghep truc tiep vao UNet chuan
    ma khong lam vo shape."""

    def __init__(self, channels, dilations=(1, 2, 4, 8)):
        super().__init__()
        branch_ch = channels // len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, branch_ch, 3, padding=d, dilation=d),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])
        self.project = nn.Conv2d(branch_ch * len(dilations), channels, 1)
        self.bn_out = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feats = torch.cat([b(x) for b in self.branches], dim=1)
        residual = self.bn_out(self.project(feats))
        return self.relu(x + residual)  # residual, KHONG thay the x


class UNetDilatedBottleneck(nn.Module):
    """Drop-in thay the archs.UNet. Interface forward(input, mode) GIONG HET,
    de dung chung voi moi ham hien co (consistency_loss, selective_merge_
    pseudo_label, DualPairRollback, ema_update, ...) khong can sua gi."""

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # DIEM KHAC BIET KIEN TRUC THAT SU - khong co trong archs.UNet.
        # v2: DAT O CA x3_0 (truoc khi pool xuong bottleneck) LAN x4_0, khong
        # chi rieng bottleneck. Ly do (rut ra tu ket qua do thuc te lan v1):
        # U-Net co skip connection rat manh - neu chi khac o x4_0, sai khac do
        # bi "pha loang" qua 3 tang decoder con lai, moi tang deu nhan them
        # skip tu x3_0/x2_0/x1_0 GIONG HET nhau giua 2 model. Dat them 1 diem
        # khac biet o x3_0 nghia la CA skip connection dau tien (vao conv3_1)
        # LAN duong bottleneck (qua pool(x3_0) -> conv4_0 -> aspp4) deu mang
        # thong tin khac nhau - sai khac khong con bi 1 tang decoder "an" di.
        self.aspp3 = ASPPResidual(nb_filter[3], dilations=(1, 2, 4))
        self.aspp4 = ASPPResidual(nb_filter[4], dilations=(1, 2, 4, 8))

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input, mode=None):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.aspp3(x3_0)  # << diem khac biet #1 - anh huong CA skip lan bottleneck
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.aspp4(x4_0)  # << diem khac biet #2

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        if mode == 'const':
            return output, [x1_0, x2_0, x3_0, x4_0]
        return output


def load_from_unet_checkpoint(model: UNetDilatedBottleneck, ckpt_path: str):
    """Nap trong so tu checkpoint archs.UNet chuan (vd stage1_ckpt hoac
    models/<source>/model.pth) vao UNetDilatedBottleneck. Cac key trung ten
    (conv0_0..conv0_4, final) se duoc nap; rieng `aspp.*` la moi hoan toan,
    KHONG co trong checkpoint UNet chuan nen se GIU NGUYEN random init.

    In ro so luong tensor da nap / bo qua de kiem tra minh bach (tranh loi
    im lang nhu img_id khong khop truoc day).
    """
    src_state = torch.load(ckpt_path, map_location='cpu')
    dst_state = model.state_dict()

    loaded, skipped_new, skipped_mismatch = [], [], []
    for k, v in src_state.items():
        if k not in dst_state:
            skipped_mismatch.append(k)
            continue
        if dst_state[k].shape != v.shape:
            skipped_mismatch.append(k)
            continue
        dst_state[k] = v
        loaded.append(k)

    new_keys = [k for k in dst_state.keys() if k not in src_state]
    model.load_state_dict(dst_state)

    print(f"[UNetDilatedBottleneck] Da nap {len(loaded)} tensor tu {ckpt_path}")
    print(f"[UNetDilatedBottleneck] Moi hoan toan (random init, thuoc ASPP): "
          f"{len(new_keys)} tensor -> {sorted(set(k.split('.')[0] for k in new_keys))}")
    if skipped_mismatch:
        print(f"[UNetDilatedBottleneck] CANH BAO bo qua {len(skipped_mismatch)} "
              f"tensor do lech ten/shape: {skipped_mismatch[:5]}...")
    return model
