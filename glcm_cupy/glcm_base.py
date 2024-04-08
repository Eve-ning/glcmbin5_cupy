from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Sequence

from tqdm import tqdm

from glcm_cupy.conf import *
from glcm_cupy.kernel import get_glcm_module
from glcm_cupy.utils import calc_grid_size, normalize_features, binner, \
    nan_to_num

FEATURES = Features.HOMOGENEITY, \
           Features.CONTRAST, \
           Features.ASM, \
           Features.MEAN, \
           Features.VARIANCE, \
           Features.CORRELATION, \
           Features.DISSIMILARITY, \
           Features.ENTROPY


@dataclass
class GLCMBase:
    """ Initialize Settings for GLCM

    Notes:
        features is a set of named integers, defined in glcm_cupy.conf

    Examples:
        To scale down the image from a 128 max value to 32, we use
        bin_from = 128, bin_to = 32.

        The range will collapse from 128 to 32.

        This thus optimizes the GLCM speed.

    Args:
        step_size: Stride Between GLCMs
        radius: Radius of Window
        bin_from: Binarize from.
        bin_to: Binarize to.
        directions: Directions to pair the windows.
        max_partition_size: Maximum number of windows to parse at once
        max_threads: Maximum number of threads to use per block
        features: Select features to be included
        normalize_features: Whether to normalize features to [0, 1]
        verbose: Whether to enable TQDM logging
    """
    radius: int = 2
    bin_from: int = 256
    bin_to: int = 256
    max_partition_size: int = MAX_PARTITION_SIZE
    max_threads: int = MAX_THREADS
    features: Sequence[int] = (Features.HOMOGENEITY,
                               Features.CONTRAST,
                               Features.ASM,
                               Features.MEAN,
                               Features.VARIANCE,
                               Features.CORRELATION,
                               Features.DISSIMILARITY,
                               Features.ENTROPY)
    normalized_features: bool = True
    verbose: bool = True

    ar_glcm: cp.ndarray = field(init=False)
    ar_features: cp.ndarray = field(init=False)
    progress: tqdm = field(init=False)
    batches: int = field(init=False, default=1)

    def __post_init__(self):
        if not self.features:
            raise ValueError(f"Features cannot be Empty")

        if not all(f in FEATURES for f in self.features):
            raise ValueError(
                f"{[f for f in self.features if f not in FEATURES]} "
                f"are not supported features."
            )

        self.ar_glcm = cp.zeros(
            (self.max_partition_size, self.bin_to, self.bin_to),
            dtype=cp.float32
        )
        self.ar_features = cp.zeros(
            (self.max_partition_size, NO_OF_FEATURES),
            dtype=cp.float32
        )
        if self.radius < 0:
            raise ValueError(f"Radius {self.radius} should be >= 0)")

        if self.bin_to not in range(2, MAX_VALUE_SUPPORTED + 1):
            raise ValueError(
                f"Target Bins {self.bin_to} must be "
                f"[2, {MAX_VALUE_SUPPORTED}]. "
            )

        module = get_glcm_module(
            homogeneity=Features.HOMOGENEITY in self.features,
            contrast=Features.CONTRAST in self.features,
            asm=Features.ASM in self.features,
            mean=Features.MEAN in self.features,
            variance=Features.VARIANCE in self.features,
            correlation=Features.CORRELATION in self.features,
            dissimilarity=Features.DISSIMILARITY in self.features,
            entropy=Features.ENTROPY in self.features
        )
        self.glcm_create_kernel = module.get_function('glcmCreateKernel')
        self.glcm_feature_kernel_0 = module.get_function('glcmFeatureKernel0')
        self.glcm_feature_kernel_1 = module.get_function('glcmFeatureKernel1')
        self.glcm_feature_kernel_2 = module.get_function('glcmFeatureKernel2')

    @property
    def _diameter(self):
        return self.radius * 2 + 1

    @abstractmethod
    def glcm_cells(self, im: cp.ndarray) -> float:
        """ Total number of GLCM Cells"""
        ...

    @property
    def nan_replacement(self):
        """ Replacement Value for NaN """
        return self.bin_to

    def run(self, im: ndarray):
        """ Executes running GLCM. Returns the GLCM Feature array

        Args:
            im: 3D/4D Image to process. Must be of shape
                ([Batch], Height, Width, Channels)

        Returns:
            An np.ndarray or cp.ndarray of Shape 4D/5D:
                ([Batch], Height, Width, Channels, GLCM Features),

        """
        was_numpy = False
        if isinstance(im, np.ndarray):
            im = cp.array(im)
            was_numpy = True

        if cp.nanmax(im) >= self.bin_from:
            raise ValueError(
                f"Found bad value ({cp.nanmax(im)}) >= bins {self.bin_from}."
            )

        if im.ndim == 2:
            raise ValueError(
                "Must be 3D. If ar.shape == (Height, Width), "
                "use ar[...,np.newaxis] to add the channel dimension."
            )
        elif im.ndim == 4:
            _ = self._run_batch(im)
            return _.get() if was_numpy else _
        elif im.ndim != 3:
            raise ValueError("Only 3D/4D images allowed.")
        self.progress = tqdm(total=self.glcm_cells(im),
                             desc="GLCM Progress",
                             unit=" Cells",
                             unit_scale=True,
                             disable=not self.verbose)

        # NaNs are replaced with the special bin_from value as uint doesn't
        # support NaN.
        im = nan_to_num(im, self.nan_replacement)
        im = binner(im, self.bin_from, self.bin_to)
        _ = self._from_im(im)
        return _.get() if was_numpy else _

    def _run_batch(self, im: ndarray):
        """ Run as a batch instead of separately.

        Notes:
            Transforms the batch axis to concat on channel axis.
            E.g. Shape = (2, H, W, 3)
            Batch 1: Channel 1, 2, 3
            Batch 2: Channel 1, 2, 3
            Transforms -> (H, W, 6)
            Channel: B1C1 B1C2 B1C3 B2C1 B2C2 B2C3
        """
        batches, *im_chn_shape, _ = im.shape
        glcm_shape = self.glcm_shape(im_chn_shape)
        batch_shape = (*glcm_shape, batches, -1, NO_OF_FEATURES)
        g = self.run(cp.concatenate(im, axis=-1))
        r = g.reshape(batch_shape)
        a = cp.moveaxis(r, 2, 0)
        return a

    @abstractmethod
    def _from_im(self, im: cp.ndarray) -> ndarray:
        """ Generates the GLCM from a multi band image

        Args:
            im: A 3 dim image as an ndarray

        Returns:
            The GLCM Array 4dim with shape
                rows, cols, channel, feature
        """

        ...

    @abstractmethod
    def make_windows(self, im_chn: cp.ndarray) -> List[
        Tuple[cp.ndarray, cp.ndarray]]:
        ...

    @abstractmethod
    def glcm_shape(self, im_chn_shape: cp.ndarray) -> Tuple:
        ...

    def _from_channel(self, im_chn: cp.ndarray) -> cp.ndarray:
        """ Generates the GLCM from an image channel

        Returns:
            The GLCM Array 3dim with shape rows, cols, feature
        """

        glcm_h, glcm_w, *_ = self.glcm_shape(im_chn.shape)
        glcm_features = [
            self.glcm_window_ij(i, j)
                .reshape(glcm_h, glcm_w, NO_OF_FEATURES)
            for i, j in self.make_windows(im_chn)
        ]

        ar = cp.stack(glcm_features).mean(axis=0)

        return normalize_features(ar, self.bin_to) \
            if self.normalized_features else ar

    def glcm_window_ij(self, windows_i: cp.ndarray,
                       windows_j: cp.ndarray):
        windows_count = windows_i.shape[0]
        glcm_features = cp.zeros(
            (windows_count, NO_OF_FEATURES),
            dtype=cp.float32
        )

        # If 45000 windows, 10000 partition size,
        # We have 5 partitions (10K, 10K, 10K, 10K, 5K)
        partition_count = math.ceil(windows_count / self.max_partition_size)

        for partition in range(partition_count):
            with cp.cuda.Device() as dev:
                # As above, we have an uneven partition
                # Though [1,2,3][:5] == [1,2,3]
                # This means windows_part_i will be correctly partitioned.
                start = partition * self.max_partition_size
                end = (partition + 1) * self.max_partition_size
                part_i = windows_i[start:end]
                part_j = windows_j[start:end]

                # We don't need to figure out the leftover partition size
                # We can simply yield the length of the leftover partition
                windows_part_count = part_i.shape[0]

                glcm_features[start:start + windows_part_count] = \
                    self.glcm_ij(part_i, part_j)
                dev.synchronize()
                self.progress.update(windows_part_count)

        return glcm_features

    def glcm_ij(self,
                i: cp.ndarray,
                j: cp.ndarray):
        """ GLCM from I J

        Examples:

            >>> ar_0 = np.random.randint(0, 100, 10, dtype=np.uint8)
            >>> ar_1 = np.random.randint(0, 100, 10, dtype=np.uint8)
            >>> g = GLCMBase().glcm_ij(ar_0[...,np.newaxis],
            ...                        ar_1[...,np.newaxis])

        Notes:
            i must be the same shape as j
            i.shape, j.shape should be [partition_size, window_size]

            For example, if you want to process 100 windows at once,
            each window with 25 cells, you should have
            i.shape == j.shape == (100, 25)

        Args:
            i: 1st np.ndarray or cp.ndarray
            j: 2nd np.ndarray or cp.ndarray

        Returns:
            The GLCM feature array, of size [partition_size, 8]

        """
        if i.ndim != 2 or j.ndim != 2:
            raise ValueError(
                f"The input dimensions must be 2. "
                f"i.ndim=={i.ndim}, j.ndim=={j.ndim}. "
                "The 1st Dim is the partitioned windows flattened, "
                "The 2nd is the window cells flattened"
            )

        if i.shape != j.shape:
            raise ValueError(f"Shape of i {i.shape} != j {j.shape}")

        # Reset data
        self.ar_glcm[:] = 0
        self.ar_features[:] = 0

        no_of_windows = i.shape[0]
        no_of_values = self._diameter ** 2

        if i.dtype != cp.uint8 or j.dtype != cp.uint8:
            raise ValueError(
                f"Image dtype must be cp.uint8 "
                f"i: {i.dtype} j: {j.dtype}"
            )

        grid = calc_grid_size(no_of_windows,
                              self.bin_to,
                              self.max_threads)
        self.glcm_create_kernel(
            grid=grid,
            block=(self.max_threads,),
            args=(
                i,
                j,
                self.bin_to,
                no_of_values,
                no_of_windows,
                self.nan_replacement,
                self.ar_glcm,
                self.ar_features
            )
        )

        feature_args = dict(
            grid=grid, block=(self.max_threads,),
            args=(self.ar_glcm,
                  self.bin_to,
                  no_of_windows,
                  self.ar_features,
                  True)
        )
        self.ar_glcm /= np.nansum(
            self.ar_glcm, axis=(1, 2)
        )[..., np.newaxis, np.newaxis]
        if self.do_stage(0): self.glcm_feature_kernel_0(**feature_args)
        if self.do_stage(1): self.glcm_feature_kernel_1(**feature_args)
        if self.do_stage(2): self.glcm_feature_kernel_2(**feature_args)

        return self.ar_features[:no_of_windows]

    def do_stage(self, stage_no: int) -> bool:
        """ Determines if running the nth stage GLCM is necessary """
        if stage_no == 2:
            return Features.CORRELATION in self.features
        elif stage_no == 1:
            return Features.VARIANCE in self.features or \
                   Features.CORRELATION in self.features
        elif stage_no == 0:
            return (
                Features.HOMOGENEITY in self.features or
                Features.CONTRAST in self.features or
                Features.ASM in self.features or
                Features.MEAN in self.features or
                Features.VARIANCE in self.features or
                Features.CORRELATION in self.features or
                Features.DISSIMILARITY in self.features or
                Features.ENTROPY in self.features
            )
