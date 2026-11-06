import numpy as np
from torch.utils.data import Sampler


class CycleSampler(Sampler):
    """Yield `num_iters` indices, cycling through random permutations of range(nums)."""

    def __init__(self, nums, num_iters):
        self.nums = int(nums)
        self.num_iters = int(num_iters)

    def __iter__(self):
        out = []
        while len(out) < self.num_iters:
            out.extend(np.random.permutation(self.nums).tolist())
        return iter(out[: self.num_iters])

    def __len__(self):
        return self.num_iters