#!/usr/bin/env python3


import dask.array as da
import numpy as np

try:
    import cupy as cp
except ImportError:
    pass

from dasf.utils.types import is_dask_array


def dask_overlap(array,
                 kernel=None,
                 axes=None,
                 boundary='reflect'):
    """
    Description
    -----------
    Generate boundaries for a Dask Array using overlap function.

    Parameters
    ----------
    array : Dask Array

    Keywork Arguments
    -----------------
    kernel : tuple (len 3), operator size
    axes : dict, boundary of the overlapped array (optional)
    boundary : str, type of the boundary reflection (optional)

    Returns
    -------
    array : Dask Array
    """

    # Compute chunk size and convert if not a Dask Array
    if not is_dask_array(array):
        raise TypeError('This function accepts only Dask Arrays.')

    if kernel is not None:
        if axes is None:
            if len(kernel) == 3:
                axes = tuple(np.array(kernel) // 2)
            else:
                raise ValueError('The kernel parameter should have length=3 '
                                 f'(current={len(kernel)}')
        ret = da.overlap.overlap(array, depth=axes, boundary=boundary)

    return ret


def dask_trim_internal(array, kernel, axes=None, boundary='reflect'):
    """
    Description
    -----------
    Trim resuling Dask Array given a specified kernel size

    Parameters
    ----------
    array : Dask Array
    kernel : tuple (len 3), operator size

    Keywork Arguments
    -----------------
    axes : dict, boundary of the overlapped array (optional)
    boundary : str, type of the boundary reflection (optional)

    Returns
    -------
    array : Trimmed Dask Array
    """

    # Compute half windows and assign to dict
    if axes is None:
        if len(kernel) == 3:
            axes = tuple(np.array(kernel) // 2)
        else:
            raise ValueError('The kernel parameter should have length=3 '
                             f'(current={len(kernel)}')
    axes = {0: axes[0], 1: axes[1], 2: axes[2]}

    return da.overlap.trim_internal(array, axes=axes, boundary=boundary)
