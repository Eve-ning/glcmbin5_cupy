from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Tuple, List, Sequence

from glcm_cupy.utils import view_as_windows_cp

try:
    from cucim.skimage.util.shape import \
        view_as_windows as view_as_windows_cucim

    USE_CUCIM = True
except:
    USE_CUCIM = False

from glcm_cupy.conf import *
from glcm_cupy.glcm_base import GLCMBase


class Direction(Enum):
    EAST = 0
    SOUTH_EAST = 1
    SOUTH = 2
    SOUTH_WEST = 3


def glcm(
    im: ndarray,
    step_size: int = 1,
    radius: int = 2,
    bin_from: int = 256,
    bin_to: int = 256,
    directions: Sequence[Direction] = (Direction.EAST,
                                       Direction.SOUTH_EAST,
                                       Direction.SOUTH,
                                       Direction.SOUTH_WEST),
    max_partition_size: int = MAX_PARTITION_SIZE,
    max_threads: int = MAX_THREADS,
    features: Sequence[int] = (Features.HOMOGENEITY,
                               Features.CONTRAST,
                               Features.ASM,
                               Features.MEAN,
                               Features.VARIANCE,
                               Features.CORRELATION,
                               Features.DISSIMILARITY),
    normalized_features: bool = True,
    skip_border: bool = False,
    verbose: bool = True
) -> ndarray:
    """
    Notes:
        features is a set of named integers, defined in glcm_cupy.conf

    Examples:
        To scale image values from a 128 max value to 32, we use
        bin_from = 128, bin_to = 32.

        The range will collapse from 128 to 32.

        This optimizes GLCM speed.

    Args:
        im: Image to Process
        step_size: Stride Between GLCMs
        radius: Radius of Window
        bin_from: Binarize from.
        bin_to: Binarize to.
        directions: Directions to pair the windows.
        max_partition_size: Maximum number of windows to parse at once
        max_threads: Maximum threads for CUDA
        features: Select features to be included
        normalized_features: Whether to normalize features to [0, 1]
        skip_border: Wheter to skip border of interfacing windows. When skipping border, result is the same as a GLCM calculated on each 7x7 window, without neighbourhood information. Default is False.
        verbose: Whether to enable TQDM logging

    Returns:
        GLCM Features
    """
    return GLCM(
        radius=radius,
        bin_from=bin_from,
        bin_to=bin_to,
        max_partition_size=max_partition_size,
        max_threads=max_threads,
        features=features,
        normalized_features=normalized_features,
        step_size=step_size,
        directions=directions,
        skip_border=skip_border,
        verbose=verbose
    ).run(im)


@dataclass
class GLCM(GLCMBase):
    step_size: int = 1
    directions: Sequence[Direction] = (
        Direction.EAST,
        Direction.SOUTH_EAST,
        Direction.SOUTH,
        Direction.SOUTH_WEST
    )
    skip_border: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.step_size <= 0:
            raise ValueError(f"Step Size {step_size} should be >= 1")

    def glcm_cells(self, im: cp.ndarray) -> int:
        """ Total number of GLCM cells to process """
        return prod(self.glcm_shape(im[..., 0].shape)) * \
               len(self.directions) * \
               im.shape[-1]

    def glcm_shape(self, im_chn_shape: tuple) -> Tuple[int, int]:
        """ Get per-channel shape after GLCM """

        return (im_chn_shape[0] - 2 * self.step_size - 2 * self.radius,
                im_chn_shape[1] - 2 * self.step_size - 2 * self.radius)

    def _from_im(self, im: cp.ndarray) -> cp.ndarray:
        """ Generates the GLCM from a multichannel image

        Args:
            im: A (H, W, C) image as ndarray

        Returns:
            The GLCM Array with shape (H, W, C, F)
        """
        return cp.stack([
            self._from_channel(im[..., ch]) for ch in range(im.shape[-1])
        ], axis=2)

    def make_windows(self, im_chn: cp.ndarray) -> List[
        Tuple[cp.ndarray, cp.ndarray]]:
        """ Convert a image channel np.ndarray, to GLCM IJ windows.

        Examples:

            Input 4 x 4 image. Radius = 1. Step Size = 1.

            +-+-+-+-+         +-----+
            |       |         | +---+-+
            |       |  ---->  | |   | |
            |       |         | |   | |
            |       |         +-+---+ |
            +-+-+-+-+           +-----+      flat
              4 x 4           1 x 1 x 3 x 3  ----> 1 x 9
                              +---+   +---+
                              flat    flat

            The output will be flattened on the x, y,

            Input 5 x 5 image. Radius = 1. Step Size = 1.

            1-2-+-+-+-+         1-----+    2-----+
            3 4       |         | +---+-+  | +---+-+  3-----+    4-----+
            |         |  ---->  | |   | |  | |   | |  | +---+-+  | +---+-+
            |         |         | |   | |  | |   | |  | |   | |  | |   | |
            |         |         +-+---+ |  +-+---+ |  | |   | |  | |   | |
            |         |           +-----+    +-----+  +-+---+ |  +-+---+ |
            +-+-+-+-+-+                                 +-----+    +-----+
              4 x 4                         2 x 2 x 3 x 3 ----> 4 x 9
                                            +---+   +---+ flat
                                            flat    flat
        Args:
            im_chn: Input Image

        Returns:
            A List of IJ windows as a Tuple[I, J]
            Each with shape (Windows, Cells)
            [
                I: (Window Ix, Cell Ix), J: (Window Ix, Cell Ix) # Direction 1
                ... # Direction 2
                ... # Direction ...
            ]
        """

        if im_chn.ndim != 2:
            raise ValueError(f"Image must be 2 dimensional. "
                             f"im.ndim={im_chn.ndim}")

        glcm_h, glcm_w, *_ = self.glcm_shape(im_chn.shape)
        if glcm_h <= 0 or glcm_w <= 0:
            raise ValueError(
                f"Step Size & Diameter exceeds size for windowing. "
                f"im.shape[0] {im_chn.shape[0]} "
                f"- 2 * step_size {self.step_size} "
                f"- 2 * radius {self.radius} <= 0 or"
                f"im.shape[1] {im_chn.shape[1]} "
                f"- 2 * step_size {self.step_size} "
                f"- 2 * radius {self.radius} + 1 <= 0 was not satisfied."
            )

        if USE_CUCIM:
            ij = view_as_windows_cucim(
                im_chn, (self._diameter, self._diameter)
            )
        else:
            ij = view_as_windows_cp(im_chn, (self._diameter, self._diameter))

        ijs: List[Tuple[cp.ndarray, cp.ndarray]] = []

        for direction in self.directions:
            i, j = self.pair_windows(ij, direction=direction)

            if self.skip_border:
                i, j = self._remove_border((i, j), direction)

            i = i.reshape((-1, *i.shape[-2:])) \
                .reshape((i.shape[0] * i.shape[1], -1))
            j = j.reshape((-1, *j.shape[-2:])) \
                .reshape((j.shape[0] * j.shape[1], -1))
            ijs.append((i, j))

        return ijs

    def _remove_border(self, ij: Tuple[cp.ndarray, cp.ndarray], direction: Direction) -> Tuple[cp.ndarray, cp.ndarray]:
        if direction == Direction.EAST:
            sl = (
                slice(None, None),
                slice(None, None),
                slice(None, None),
                slice(None, -self.step_size),
            )
        elif direction == Direction.SOUTH_EAST:
            sl = (
                slice(None, None),
                slice(None, None),
                slice(None, -self.step_size),
                slice(None, -self.step_size),
            )
        elif direction == Direction.SOUTH:
            sl = (
                slice(None, None),
                slice(None, None),
                slice(None, -self.step_size),
                slice(None, None),
            )
        elif direction == Direction.SOUTH_WEST:
            sl = (
                slice(None, None),
                slice(None, None),
                slice(None, -self.step_size),
                slice(self.step_size, None),
            )
        else:
            raise ValueError("Invalid Direction")

        return ij[0][sl], ij[1][sl]

    def pair_windows(self, ij: ndarray,
                     direction: Direction) -> Tuple[ndarray, ndarray]:
        """ Pairs the ij windows in specified direction

        Notes:

            For an image:
                                          East
                           +-----------+  +-----------+
                           |           |  |           |
                           |   +---+   |  |   +---+---+
                           |   |   |   |  |   | I | J |
                           |   +---+   |  |   +---+---+
                           |           |  |           |
                           +-----------+  +-----------+
            South West     South          South East
            +-----------+  +-----------+  +-----------+
            |           |  |           |  |           |
            |   +---+   |  |   +---+   |  |   +---+   |
            |   | I |   |  |   | I |   |  |   | I |   |
            +---+---+   |  |   +---+   |  |   +---+---+
            | J |       |  |   | J |   |  |       | J |
            +---+-------+  +---+---+---+  +-------+---+

            i window will be the one in the middle
            j window will be the outer skirts

            Top left few pixels will be discarded.

        Args:
            ij: The ij output from make_windows
            direction: Direction to pair

        Returns:

        """
        step = self.step_size
        i = ij[step:-step, step:-step]
        if direction == Direction.EAST:
            j = ij[step:-step, step * 2:]
        elif direction == Direction.SOUTH:
            j = ij[step * 2:, step:-step]
        elif direction == Direction.SOUTH_WEST:
            j = ij[step * 2:, :-step * 2]
        elif direction == Direction.SOUTH_EAST:
            j = ij[step * 2:, step * 2:]
        else:
            raise ValueError("direction must be of class Direction.")

        return i, j
