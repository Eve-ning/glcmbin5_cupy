import numpy as np
import pytest

from glcm_cupy import GLCM
from glcm_cupy.conf import *
from glcm_cupy.glcm.glcm_py import glcm_py_ij
from tests.unit_tests.util.glcm_py_skimage import glcm_py_skimage


@pytest.mark.parametrize(
    "i",
    [
        np.asarray([0, ] * 9, dtype=np.uint8),
        np.asarray([1, ] * 9, dtype=np.uint8),
        np.asarray([255, ] * 9, dtype=np.uint8),
        np.asarray([0, 1, 2, 3, 4, 252, 253, 254, 255], dtype=np.uint8),
    ]
)
@pytest.mark.parametrize(
    "j",
    [
        np.asarray([0, ] * 9, dtype=np.uint8),
        np.asarray([1, ] * 9, dtype=np.uint8),
        np.asarray([255, ] * 9, dtype=np.uint8),
        np.asarray([0, 1, 2, 3, 4, 252, 253, 254, 255], dtype=np.uint8),
    ]
)
def test_glcm_ij(i, j):
    # We only test with 2 windows to reduce time taken.
    windows = 2
    g = GLCM(radius=1).glcm_ij(
        cp.asarray(np.tile(i, (windows, 1))),
        cp.asarray(np.tile(j, (windows, 1)))
    )

    # The sum of the values, since tiled, will be scaled by no of windows.
    actual = [
        float(g[..., Features.HOMOGENEITY].sum() / windows),
        float(g[..., Features.CONTRAST].sum() / windows),
        float(g[..., Features.ASM].sum() / windows),
        float(g[..., Features.MEAN].sum() / windows),
        float(g[..., Features.VARIANCE].sum() / windows),
        float(g[..., Features.CORRELATION].sum() / windows),
        float(g[..., Features.DISSIMILARITY].sum() / windows),
        float(g[..., Features.ENTROPY].sum() / windows)
    ]

    expected = glcm_py_ij(i, j, 256, 256)
    assert actual == pytest.approx(expected)

    # The sum of the values, since tiled, will be scaled by no of windows.
    actual_skimage = dict(
        homogeneity=float(g[..., Features.HOMOGENEITY].sum() / windows),
        contrast=float(g[..., Features.CONTRAST].sum() / windows),
        asm=float(g[..., Features.ASM].sum() / windows),
        correlation=float(g[..., Features.CORRELATION].sum() / windows),
        dissimilarity=float(g[..., Features.DISSIMILARITY].sum() / windows)
    )

    if (i == j).all():
        actual_skimage['correlation'] = 1

    expected_skimage = glcm_py_skimage(i, j)
    assert actual_skimage == pytest.approx(expected_skimage, abs=0.01)

def test_glcm_ge_bin_from():
    """ Assert raise will happen on values >= bin_from """
    with pytest.raises(ValueError):
        GLCM(bin_from=256, bin_to=256).run(np.ones((10, 10)) * 256)
    with pytest.raises(ValueError):
        GLCM(bin_from=256, bin_to=256).run(np.ones((10, 10)) * 257)
