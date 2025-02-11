#!/usr/bin/env python3

import dask.array as da
import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

from dasf.transforms import Transform


class RGBBlending(Transform):
    def __init__(self):
        super().__init__()

    def _to_grey(self, X, xp):
        max_val = xp.max(X)
        min_val = xp.min(X)

        X = ((X - min_val) / max_val) * 255

        return X.astype(int)

    def _transform(self, X, xp):
        assert len(X) == 3

        X_r, X_g, X_b = X[0], X[1], X[2]

        r_to_g = self._to_grey(X_r, xp)
        g_to_g = xp.left_shift(self._to_grey(X_g, xp), 8)
        b_to_g = xp.left_shift(self._to_grey(X_b, xp), 16)

        X = xp.zeros_like(X_r)

        X = r_to_g + g_to_g + b_to_g

        return X

    def _lazy_transform_gpu(self, X):
        return self._transform(X, da)

    def _lazy_transform_cpu(self, X):
        return self._transform(X, da)

    def _transform_gpu(self, X):
        return self._transform(X, cp)

    def _transform_cpu(self, X):
        return self._transform(X, np)
