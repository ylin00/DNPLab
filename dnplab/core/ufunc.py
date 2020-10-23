import numpy as np

from . import nddata

# Boolean, unsigned integer, signed integer, float, complex.
_NUMERIC_KINDS = set("buifc")


def is_numeric_vector(array):
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Parameters
    ----------
    array : array-like
        The array to check.

    Returns
    -------
    is_numeric_vector : `bool`
        True if the array has a numeric datatype, False if not.

    """
    return (
        np.asarray(array).dtype.kind in _NUMERIC_KINDS and np.asarray(array).ndim == 1
    )
