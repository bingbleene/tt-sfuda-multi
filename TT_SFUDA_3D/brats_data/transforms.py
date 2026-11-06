"""Minimal transforms so that the YAML strings
   Compose([Pad((0,0,0,0,0)), NumpyType((np.float32, np.int64))])
can be eval'ed. Each transform takes/returns a list [x, y]
with x: (H, W, D, C) and y: (H, W, D)."""
import numpy as np


class Identity:
    def __call__(self, data):
        return data


class Compose:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, data):
        for op in self.ops:
            data = op(data)
        return data


class Pad:
    """pad = (pad_H, pad_W, pad_D, pad_C, pad_extra) -- appended at the END of each axis.
    (0,0,0,0,0) is a no-op (what the author's configs use)."""

    def __init__(self, pad):
        self.pad = tuple(pad)

    def __call__(self, data):
        x, y = data[0], data[1]
        px = list(self.pad[: x.ndim]) + [0] * max(0, x.ndim - len(self.pad))
        if any(px):
            x = np.pad(x, [(0, p) for p in px[: x.ndim]], mode="constant")
        py = list(self.pad[: y.ndim])
        if any(py):
            y = np.pad(y, [(0, p) for p in py], mode="constant")
        return [x, y] + list(data[2:])


class NumpyType:
    def __init__(self, types):
        self.types = types

    def __call__(self, data):
        return [d.astype(t) if t is not None else d for d, t in zip(data, self.types)] + list(data[len(self.types):])