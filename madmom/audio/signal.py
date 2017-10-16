# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains basic signal processing functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

# from madmom.processors import Processor, BufferProcessor


# signal functions
def smooth(signal, kernel):
    """
    Smooth the signal along its first axis.

    Parameters
    ----------
    signal : numpy array
        Signal to be smoothed.
    kernel : numpy array or int
        Smoothing kernel (size).

    Returns
    -------
    numpy array
        Smoothed signal.

    Notes
    -----
    If `kernel` is an integer, a Hamming window of that length will be used
    as a smoothing kernel.

    """
    # check if a kernel is given
    if kernel is None:
        return signal
    # size for the smoothing kernel is given
    elif isinstance(kernel, (int, np.integer)):
        if kernel == 0:
            return signal
        elif kernel > 1:
            # use a Hamming window of given length
            kernel = np.hamming(kernel)
        else:
            raise ValueError("can't create a smoothing kernel of size %d" %
                             kernel)
    # otherwise use the given smoothing kernel directly
    elif isinstance(kernel, np.ndarray):
        kernel = kernel
    else:
        raise ValueError("can't smooth signal with %s" % kernel)
    # convolve with the kernel and return
    if signal.ndim == 1:
        return np.convolve(signal, kernel, 'same')
    elif signal.ndim == 2:
        from scipy.signal import convolve2d
        return convolve2d(signal, kernel[:, np.newaxis], 'same')
    else:
        raise ValueError('signal must be either 1D or 2D')
