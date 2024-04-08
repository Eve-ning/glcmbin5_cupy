from dataclasses import dataclass
from typing import List

import cupy as cp
import numpy as np


@dataclass
class GLCMPyBase:
    bin_from: int
    bin_to: int
    radius: int = 2

    @property
    def diameter(self) -> int:
        return self.radius * 2 + 1

    def glcm_ij(self, i: cp.ndarray, j: cp.ndarray) -> List[float]:
        """ Get GLCM features using Python

        Notes:
            This is to assert the actual values in the tests.
            Technically can be used in production though very slow.

        Args:
            i: Window I, may be ndim
            j: Window J, may be ndim

        Returns:
            List of these features [
                homogeneity, contrast, asm, mean, mean_j,
                var_i, var_j, correlation
            ].
            Values as float.
        """

        i_flat = i.flatten()
        j_flat = j.flatten()
        assert len(i_flat) == len(j_flat), \
            f"The shapes for i {i.shape} != j {j.shape}."

        glcm = np.zeros((self.bin_to, self.bin_to), dtype=float)

        # Populate the GLCM
        for i_, j_ in zip(i_flat, j_flat):
            glcm[i_, j_] += 1
            glcm[j_, i_] += 1

        # Convert to probability
        glcm /= len(i_flat) * 2

        homogeneity = contrast = asm = mean = var = \
            correlation = dissimilarity = entropy = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                dissimilarity += glcm[i, j] * np.abs(i - j)
                homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
                contrast += glcm[i, j] * (i - j) ** 2
                asm += glcm[i, j] ** 2
                mean += glcm[i, j] * i
                if glcm[i, j] != 0:
                    entropy += glcm[i, j] * -np.log(glcm[i, j])

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                var += glcm[i, j] * (i - mean) ** 2

        # Variances cannot be 0 else ZeroDivisionError
        if var != 0:
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    correlation += glcm[i, j] * (i - mean) * (j - mean) / var

        return [homogeneity, contrast, asm, mean, var, correlation,
                dissimilarity, entropy]
