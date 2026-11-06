"""Re-implementation of the (missing) `data.datasets` module of Vibashan/tt-sfuda.

SingleData(list_file, root, for_train, return_target, transforms)
  * list_file : text file, one case name per line (e.g. BraTS19_2013_10_1)
  * root      : directory containing one folder per case
                root/<case>/<case>_{flair,t1ce,t1,t2,seg}.nii[.gz]
  * item      : x (4, H, W, D) float32, channel order [FLAIR, T1ce, T1, T2]
                y (H, W, D) int64 with RAW BraTS labels {0,1,2,4}  (dice() relies on label 4)
  * collate   : default_collate  ->  x (B,4,H,W,D), y (B,H,W,D)

Normalisation (ASSUMPTION - the author's loader is not public):
  DMFNet-style, per modality, on non-zero (brain) voxels: clip to [0.2, 99.8] percentile,
  then z-score. Background stays 0. Set NORMALIZE = 'zscore' to skip the clipping,
  or 'none' to feed raw intensities.
"""
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from .transforms import *  # noqa: F401,F403  (needed by eval of YAML transform strings)

MODALITIES = ("flair", "t1ce", "t1", "t2")  # channel order expected by tt_sfuda_3d.py
NORMALIZE = "clip_zscore"  # 'clip_zscore' | 'zscore' | 'none'
CLIP_PCT = (0.2, 99.8)


def _find(case_dir, case, suffix):
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(case_dir, f"{case}_{suffix}{ext}")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{case_dir}/{case}_{suffix}(.nii|.nii.gz)")


def normalize_volume(v, mode=NORMALIZE):
    """v: (H,W,D) float32. Returns normalised copy; zero voxels stay zero."""
    v = v.astype(np.float32, copy=True)
    if mode == "none":
        return v
    mask = v > 0
    if not mask.any():
        return v
    vals = v[mask]
    if mode == "clip_zscore":
        lo, hi = np.percentile(vals, CLIP_PCT)
        v[mask & (v < lo)] = lo
        v[mask & (v > hi)] = hi
        vals = v[mask]
    mean, std = vals.mean(), vals.std()
    v[mask] = (vals - mean) / (std + 1e-8)
    return v


class SingleData(Dataset):
    def __init__(self, list_file, root="", for_train=False, return_target=True, transforms=None):
        with open(list_file) as f:
            self.names = [ln.strip() for ln in f if ln.strip()]
        self.root = root
        self.for_train = for_train
        self.return_target = return_target
        self.transforms = eval(transforms or "Identity()")

    def __len__(self):
        return len(self.names)

    def load_case(self, name):
        d = os.path.join(self.root, name)
        chans = []
        for m in MODALITIES:
            arr = np.asarray(nib.load(_find(d, name, m)).dataobj, dtype=np.float32)
            chans.append(normalize_volume(arr))
        x = np.stack(chans, axis=-1)  # (H,W,D,4)
        y = None
        if self.return_target:
            y = np.asarray(nib.load(_find(d, name, "seg")).dataobj).astype(np.int64)  # raw labels
        return x, y

    def __getitem__(self, index):
        x, y = self.load_case(self.names[index])
        if y is None:  # no target requested: transforms still expect a 2-list
            y = np.zeros(x.shape[:3], dtype=np.int64)
        x, y = self.transforms([x, y])[:2]
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(3, 0, 1, 2)))  # (4,H,W,D)
        y = torch.from_numpy(np.ascontiguousarray(y))
        return x, y

    def collate(self, batch):
        return default_collate(batch)