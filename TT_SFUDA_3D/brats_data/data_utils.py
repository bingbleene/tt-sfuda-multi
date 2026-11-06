import random

import numpy as np
import torch


def init_fn(worker_id):
    """DataLoader worker_init_fn: different but reproducible seed per worker."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def add_mask(x, mask, dim=1):
    """Append `mask` as an extra channel of x along `dim` (kept for interface compatibility;
    unused by TT-SFUDA because the loader returns only (x, target))."""
    mask = mask.unsqueeze(dim)
    return torch.cat([x, mask.to(x.dtype)], dim=dim)