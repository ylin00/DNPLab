import numpy as np
import pytest

from dnplab.core.ufunc import is_numeric_vector


@pytest.mark.parametrize("numeric_item", [True, 1, -1, 1.0, 1 + 1j])
def test_is_numeric_vector(numeric_item):
    for y in ([numeric_item], [numeric_item, numeric_item]):
        for z in (y, np.array(y)):
            assert is_numeric_vector(z)


@pytest.mark.parametrize("scalar", [1, np.array(1)])
def test_scalar_not_numeric_vector(scalar):
    assert not is_numeric_vector(scalar)


@pytest.mark.parametrize(
    "high_d", [[[1]], [np.array([1])], np.array([[1]]), np.array([np.array([1])])]
)
def test_2d_not_numeric_vector(high_d):
    assert not is_numeric_vector(high_d)


@pytest.mark.parametrize("not_a_num", [object(), "string", u"unicode", None])
def test_not_numeric_vector(not_a_num):
    for y in (not_a_num, [not_a_num], [not_a_num, not_a_num]):
        for z in (y, np.array(y)):
            assert not is_numeric_vector(z)
