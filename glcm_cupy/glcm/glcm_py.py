from dataclasses import dataclass

import cupy as cp
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from typing import Tuple, Dict

from glcm_cupy.conf import NO_OF_FEATURES, ndarray
from glcm_cupy.glcm_py_base import GLCMPyBase
from glcm_cupy.utils import normalize_features


def glcm_py_im(ar: ndarray, bin_from: int, bin_to: int,
               radius: int = 2,
               step: int = 1,
               skip_border = False):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to,
                  radius=radius,
                  step=step,
                  skip_border=skip_border).glcm_im(ar)


def glcm_py_chn(ar: cp.ndarray,
                bin_from: int,
                bin_to: int,
                radius: int = 2,
                step: int = 1,
                skip_border = False):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to,
                  radius=radius,
                  step=step,
                  skip_border=skip_border).glcm_chn(ar)


def glcm_py_ij(i: cp.ndarray,
               j: cp.ndarray,
               bin_from: int, bin_to: int):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to).glcm_ij(i, j)


@dataclass
class GLCMPy(GLCMPyBase):
    step: int = 1
    skip_border: bool = False

    def glcm_chn(self, ar: cp.ndarray):

        ar = (ar / self.bin_from * self.bin_to).astype(cp.uint8)
        ar_w = sliding_window_view(ar, (self.diameter, self.diameter))
        def windows(ar: ndarray):
            return ar.reshape((-1, self.diameter, self.diameter))
            
        def flat(ar: ndarray):
            return ar.reshape((ar.shape[0], -1))

        ar_w_i = windows(ar_w[self.step:-self.step, self.step:-self.step])
        ar_w_j_sw = windows(ar_w[self.step * 2:, :-self.step * 2])
        ar_w_j_s = windows(ar_w[self.step * 2:, self.step:-self.step])
        ar_w_j_se = windows(ar_w[self.step * 2:, self.step * 2:])
        ar_w_j_e = windows(ar_w[self.step:-self.step, self.step * 2:])

        feature_ar = np.zeros((ar_w_i.shape[0], 4, NO_OF_FEATURES))

        for j_e, ar_w_j in enumerate(
            (ar_w_j_sw, ar_w_j_s, ar_w_j_se, ar_w_j_e)):
            ar_i, ar_j = ar_w_i, ar_w_j
            if self.skip_border:
                ar_i, ar_j = self._remove_border((ar_i, ar_j), j_e)
            ar_i = flat(ar_i)
            ar_j = flat(ar_j)
            for e, (i, j) in tqdm(enumerate(zip(ar_i, ar_j)),
                                  total=ar_i.shape[0]):
                
                feature_ar[e, j_e] = self.glcm_ij(i, j)

        h, w = ar_w.shape[:2]
        feature_ar = feature_ar.mean(axis=1)

        feature_ar = feature_ar.reshape(
            (h - self.step * 2, w - self.step * 2, NO_OF_FEATURES))

        return normalize_features(feature_ar, self.bin_to)

    def _remove_border(self, ij: Tuple[cp.ndarray, cp.ndarray], direction: int) -> Tuple[cp.ndarray, cp.ndarray]:
        if direction == 0: # Direction.SOUTH_WEST
            sl = (
                slice(None, None),
                slice(None, -self.step),
                slice(self.step, None),
            )
        elif direction == 1: # Direction.SOUTH
            sl = (
                slice(None, None),
                slice(None, -self.step),
                slice(None, None),
            )
        elif direction == 2: # Direction.SOUTH_EAST
            sl = (
                slice(None, None),
                slice(None, -self.step),
                slice(None, -self.step),
            )
        elif direction == 3:  # Direction.EAST
            sl = (
                slice(None, None),
                slice(None, None),
                slice(None, -self.step),
            )
        else:
            raise ValueError("Invalid Direction")

        return ij[0][sl], ij[1][sl]

    def glcm_im(self, ar: ndarray):
        was_cupy = False
        if isinstance(ar, cp.ndarray):
            ar = np.array(ar)
            was_cupy = True
        _ = np.stack([self.glcm_chn(ar[..., ch])
                      for ch
                      in range(ar.shape[-1])], axis=2)
        return cp.array(_) if was_cupy else _
