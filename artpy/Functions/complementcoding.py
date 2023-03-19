"""
    SalientTopoART PY: A Python library of Salient TopoART.

    This function implements complement coding and scaling for a
    given input
"""
__version__ = "0.1"
__author__ = "Raghu Yelugam"

import numpy as np


def ComplementCoding(data: np.ndarray,
                     dim: int = 0) -> np.ndarray:

    """
    :param data: input data
    :param dim: dimension along with normalisation should be done
    :return CCData: scaled complement coded input
    """

    if isinstance(data, np.ndarray):
        dim_max = np.max(data, axis=dim)
        dim_min = np.min(data, axis=dim)

        if dim == 0:
            normalised = (data - dim_min)/(dim_max - dim_min + 1e-8)
        if dim == 1:
            normalised = ((data.T - dim_min)/(dim_max - dim_min + 1e-8)).T

        CCData = np.concatenate((normalised, 1-normalised), axis=1-dim)
    else:
        raise TypeError("Not either a list or an np.ndarray")

    return CCData
