import numpy as np
import pytest

from glcm_cupy import GLCM, glcm
from glcm_cupy.glcm.glcm_py import glcm_py_im


@pytest.mark.parametrize(
    "ar",
    [
        np.ones([16, 16, 3]) * 31,
        np.zeros([16, 16, 3])
    ]
)
def test_glcm_normalize(ar):
    """ Assert that our normalize will reduce the range to 0, 1 """
    g = GLCM(radius=3, bin_from=32, bin_to=32,
             normalized_features=False).run(ar)
    assert g.max() >= 1
    assert g.min() >= 0
    g = GLCM(radius=3, bin_from=32, bin_to=32,
             normalized_features=True).run(ar)
    assert g.min() >= 0
    assert g.max() <= 1


@pytest.mark.parametrize(
    "size",
    [15, ]
)
@pytest.mark.parametrize(
    "bin_from",
    [4, 16]
)
@pytest.mark.parametrize(
    "bin_to",
    [4, 16]
)
@pytest.mark.parametrize(
    "radius",
    [1, 2, 4]
)
@pytest.mark.parametrize(
    "skip_border",
    [False, True]
)
def test_glcm(size, bin_from, bin_to, radius, skip_border):
    ar = np.random.randint(0, bin_from, [size, size, 1])
    g = GLCM(radius=radius, bin_from=bin_from, bin_to=bin_to, skip_border=skip_border).run(ar)
    g_fn = glcm(ar, radius=radius, bin_from=bin_from, bin_to=bin_to, skip_border=skip_border)
    expected = glcm_py_im(ar, radius=radius, bin_from=bin_from, bin_to=bin_to, skip_border=skip_border)
    assert g == pytest.approx(expected, abs=0.001)
    assert g_fn == pytest.approx(expected, abs=0.001)


def test_channel_independence():
    """ This asserts that the channel GLCMs are independent """
    ar = np.random.randint(0, 255, [16, 16, 2])
    radius, bins = 3, 16
    ar0, ar1 = ar[..., 0:1], ar[..., 1:2]
    glcm = GLCM(radius=radius, bin_from=256, bin_to=bins)
    g = glcm.run(ar)
    g0 = glcm.run(ar0)
    g1 = glcm.run(ar1)
    g_exp = np.concatenate([g0, g1], axis=2)
    assert g == pytest.approx(g_exp, abs=1e-05)
