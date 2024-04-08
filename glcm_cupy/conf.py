from enum import IntEnum
from pathlib import Path
from typing import Union

import cupy as cp
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.absolute()

MAX_VALUE_SUPPORTED = 256
NO_OF_VALUES_SUPPORTED = 256 ** 2

MAX_THREADS = 512  # Min. Maximum supported threads.

# For a 1000 x 256 x 256 GLCM, you need ~ 262MB of memory.
# Each cell is 4 bytes (float32)
MAX_PARTITION_SIZE = 1000


class Features(IntEnum):
    HOMOGENEITY = 0
    CONTRAST = 1
    ASM = 2
    MEAN = 3
    VARIANCE = 4
    CORRELATION = 5
    DISSIMILARITY = 6
    ENTROPY = 7


NO_OF_FEATURES = len(Features)

ndarray = Union[np.ndarray, cp.ndarray]
