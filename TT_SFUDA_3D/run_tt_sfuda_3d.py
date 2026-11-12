"""Launcher for TT-SFUDA 3D that leaves tt_sfuda_3d.py / utils/ untouched.

It imports the original module, re-uses its functions (sfuda_target, sfuda_task, validate, ...) and only
re-implements `main()` so that
  * paths come from the command line (no hard-coded /media/vibsss/... paths),
  * several evaluation lists can be scored in ONE adaptation run
    (source-only 'direct' score before adaptation, 'tt' score after Stage I + II),
  * results are written to a json file.

Example:
  python run_tt_sfuda_3d.py --cfg unet_t2 --data_dir /kaggle/working/data/train/all \
      --train_dir /kaggle/working/ckpts --valid_list val_s2001.txt,all.txt --out_json res/unet_t2.json
"""
import argparse
import json
import logging
import os
import re
import sys
import time

import yaml

# utils/parser.py calls yaml.load(f) without Loader -> error on PyYAML>=6. Patch here, not in the repo.
_yaml_load = yaml.load
yaml.load = lambda stream, Loader=None, **kw: _yaml_load(stream, Loader=Loader or yaml.FullLoader, **kw)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# tt_sfuda_3d.py gốc import `data`; package của mình tên `brats_data` (thư mục `data/` bị .gitignore) nên đăng ký lại dưới tên `data`
import importlib
sys.modules['data'] = importlib.import_module('brats_data')
for _sub in ('datasets', 'sampler', 'data_utils', 'transforms'):
    sys.modules['data.' + _sub] = importlib.import_module('brats_data.' + _sub)
import torch                                   # noqa: E402
import torch.optim as optim                    # noqa: E402
from torch.utils.data import DataLoader        # noqa: E402

import tt_sfuda_3d as T                        # noqa: E402  (original, unmodified)
from utils import Parser                       # noqa: E402

TGT_NUM = {"unet_t1": 1, "unet_t2": 0, "unet_flair": 3}   # same mapping as tt_sfuda_3d.main()
_PAT = re.compile(r"Adapt to \w+\s+whole: ([\d.]+), core: ([\d.]+), enhancing: ([\d.]+)")


class _Grab(logging.Handler):
    """Keeps the last 'Adapt to ... whole/core/enhancing' line that validate() logs."""
    def __init__(self):
        super().__init__()
        self.last = None

    def emit(self, record):
        m = _PAT.search(record.getMessage())
        if m:
            self.last = [100 * float(v) for v in m.groups()]

def _rng_get():
    return torch.get_rng_state(), (torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def _rng_set(s):
    torch.set_rng_state(s[0])
    if s[1] is not None:
        torch.cuda.set_rng_state(s[1])


def chunked_backward(model, aug_input, ps_output, chunk=1):
    """Same loss as tt_sfuda_3d.sfuda_target but backward `chunk` volumes at a time (~1/5 memory).
    Pass 1 (no grad) gets the pooled Dice sums A, B; pass 2 backprops each chunk with A, B fixed, so the
    summed gradient equals the full-batch gradient. Dropout masks are replayed via the saved RNG state."""
    eps = T.criterions.eps
    n = aug_input.shape[0]
    t_c = [(ps_output == c).float() for c in range(1, 5)]
    chunks = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    dev = ps_output.device
    A = torch.zeros(4, device=dev); B = torch.zeros(4, device=dev)
    states = []
    with torch.no_grad():
        for s, e in chunks:
            states.append(_rng_get())
            p = torch.softmax(model(aug_input[s:e].cuda()), dim=1)
            for c in range(4):
                A[c] += (p[:, c + 1] * t_c[c]).sum()
                B[c] += p[:, c + 1].sum()
    B = B + torch.stack([t.sum() for t in t_c]) * n
    den = B + eps
    dice_sum = (1.0 - (2 * A + eps) / den).sum()
    coef_a, coef_b = 2.0 / den, (2 * A + eps) / den ** 2
    ce_tot = ent_tot = 0.0
    for k, (s, e) in enumerate(chunks):
        _rng_set(states[k])
        m = e - s
        out = model(aug_input[s:e].cuda())
        ce = torch.nn.functional.cross_entropy(out, ps_output.repeat(m, 1, 1, 1))
        ent = T.softmax_entropy_loss(out)
        p = torch.softmax(out, dim=1)
        dterm = sum(-coef_a[c] * (p[:, c + 1] * t_c[c]).sum() + coef_b[c] * p[:, c + 1].sum() for c in range(4))
        (0.01 * (ce * m / n + 0.25 * dterm) + ent * m / n).backward()
        ce_tot += ce.item() * m / n
        ent_tot += ent.item() * m / n
    return 0.01 * (ce_tot + 0.25 * dice_sum.item()) + ent_tot


def sfuda_target_chunked(train_loader, pseduo_model, msrc_model, criterion, optimizer, break_pt, tgt_num, chunk=1):
    """Drop-in replacement of tt_sfuda_3d.sfuda_target (Stage I) with low memory."""
    assert criterion is T.criterions.cross_entropy_dice, "chunked Stage I is written for cross_entropy_dice"
    meter = T.AverageMeter()
    pseduo_model.eval(); msrc_model.cuda(); msrc_model.train()
    pbar = T.tqdm(total=break_pt)
    for i, data in enumerate(train_loader):
        data = [t.cuda(non_blocking=True) for t in data]
        x = data[0]
        input = x[:, tgt_num, :, :, :]
        aug_input = T.build_pseduo_augmentation(input.cpu().numpy())
        with torch.no_grad():
            aug_output = pseduo_model(aug_input.cuda())
            ps_output = T.uncert_voting(aug_output.detach())
        del aug_output
        optimizer.zero_grad()
        loss = chunked_backward(msrc_model, aug_input, ps_output, chunk)
        optimizer.step()
        meter.update(loss, input.size(0))
        pbar.set_postfix(T.OrderedDict([('loss', meter.avg)]))
        pbar.update(1)
        if i == break_pt:
            break
    pbar.close()
    return T.OrderedDict([('loss', meter.avg)])

def run(args, valid_lists, grab):
    tgt_num = TGT_NUM[args.cfg]
    model_file = os.path.join(args.getdir(), args.ckpt)

    msrc_model = T.models.unet.Unet_wo()
    msrc_model.load_state_dict(torch.load(model_file)['state_dict'])
    msrc_model = msrc_model.cuda()
    msrc_model.train()

    tgt_model = T.models.unet.Unet_wo()
    src_optimizer = optim.Adam(filter(lambda p: p.requires_grad, msrc_model.parameters()),
                               lr=args.opt_params.lr, weight_decay=0.0001)
    tgt_optimizer = optim.Adam(filter(lambda p: p.requires_grad, tgt_model.parameters()),
                               lr=args.opt_params.lr, weight_decay=0.0001)

    pseudo_model = T.models.unet.Unet_wo()
    pseudo_model.load_state_dict(msrc_model.state_dict())
    pseudo_model.cuda()
    pseudo_model.eval()

    Dataset = getattr(T.datasets, args.dataset)
    train_set = Dataset(os.path.join(args.data_dir, args.train_list), root=args.data_dir,
                        for_train=True, transforms=args.train_transforms)
    break_pt = 25
    num_iters = (len(train_set) * 1) // args.batch_size
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=train_set.collate,
                              sampler=T.CycleSampler(len(train_set), num_iters * args.batch_size),
                              num_workers=args.workers, pin_memory=True, worker_init_fn=T.init_fn)

    valid = {}
    for name in valid_lists:
        vs = Dataset(os.path.join(args.data_dir, name), root=args.data_dir, for_train=False,
                     return_target=True, transforms=args.test_transforms)
        valid[os.path.splitext(name)[0]] = (DataLoader(vs, batch_size=1, shuffle=False,
                                                       collate_fn=vs.collate, num_workers=4,
                                                       pin_memory=True), vs)
    criterion = getattr(T.criterions, args.criterion)
    res = {"direct": {}, "tt": {}}

    def evaluate(model, key):
        for tag, (vl, vs) in valid.items():
            grab.last = None
            T.validate(vl, model, '', vs.names, tgt_num=tgt_num, scoring=True, mode=key)
            res[key][tag] = grab.last
            print(f"[{key}] {tag}: WT/TC/ET = " + " / ".join(f"{v:.2f}" for v in grab.last), flush=True)

    print("Source-only (Direct Testing) ...", flush=True)
    evaluate(pseudo_model, "direct")
    if args.direct_only:
          return res

    print("Stage I ...", flush=True)
    t0 = time.time()
    for _ in range(args.stage1):
        if args.chunk > 0:
            sfuda_target_chunked(train_loader, pseudo_model, msrc_model, criterion, src_optimizer, break_pt, tgt_num, args.chunk)
        else:
            T.sfuda_target(train_loader, pseudo_model, msrc_model, criterion, src_optimizer, break_pt, tgt_num)
        torch.cuda.empty_cache()

    msrc_model.eval()
    tgt_model.load_state_dict(msrc_model.state_dict())
    tgt_model.cuda()
    tgt_model.train()

    print("Stage II ...", flush=True)
    for _ in range(args.stage2):
        T.sfuda_task(train_loader, msrc_model, tgt_model, criterion, tgt_optimizer, break_pt, tgt_num)
        torch.cuda.empty_cache()

    print("TT-SFUDA evaluation ...", flush=True)
    evaluate(msrc_model, "tt")
    res["minutes"] = (time.time() - t0) / 60
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, choices=list(TGT_NUM))
    ap.add_argument('--data_dir', required=True, help='folder with all.txt, val*.txt and one sub-folder per case')
    ap.add_argument('--train_dir', required=True, help='folder containing <cfg>/source_model.tar')
    ap.add_argument('--valid_list', default='val.txt', help='comma separated list files inside data_dir')
    ap.add_argument('--stage2', type=int, default=None, help='Stage-II epochs (repo default: 2, paper: 10)')
    ap.add_argument('--keep_rate', type=float, default=None, help='EMA keep rate (repo default: 0.9, paper: 0.99)')
    ap.add_argument('--out_json', default=None)
    ap.add_argument('--lr', type=float, default=None, help='learning rate (repo yaml: 1e-5, paper: 1e-4)')
    ap.add_argument('--chunk', type=int, default=1, help='Stage I: so khoi moi lan backward (1 = vua GPU 15GB; 0 = goc, batch 5)')
    ap.add_argument('--direct_only', action='store_true', help='chi cham source-only, khong adaptation')
    cli = ap.parse_args()

    args = Parser(cli.cfg, log='test').add_args(argparse.Namespace(cfg=cli.cfg))
    args.data_dir, args.train_dir = cli.data_dir, cli.train_dir
    args.ckpt = 'source_model.tar'
    args.chunk = cli.chunk
    args.direct_only = cli.direct_only
    if cli.stage2 is not None:
        args.stage2 = cli.stage2
    if cli.lr is not None:
        args.opt_params.lr = cli.lr
    if cli.keep_rate is not None:   # sfuda_task() hard-codes 0.9; override via the module attribute
        _utm = T.update_teacher_model
        T.update_teacher_model = lambda s, t, keep_rate=None: _utm(s, t, keep_rate=cli.keep_rate)

    grab = _Grab()
    logging.getLogger('').addHandler(grab)
    result = run(args, [x.strip() for x in cli.valid_list.split(',')], grab)
    if cli.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(cli.out_json)), exist_ok=True)
        json.dump(result, open(cli.out_json, 'w'), indent=1)