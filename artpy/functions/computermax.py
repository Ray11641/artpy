"""
    ARTPY: A Python library of Adaptive Resonance Theory based learning
     models.

     This file provides computermax function.
"""
import typing
import numpy as np
from . import *

def computermax(input: Union[np.ndarray, List[np.ndarray]]) -> float:
    if isinstance(input, list):
        input = np.array(input)

    (nObs,nDim) = input.shape
    RMax = 0
    for i in range(nObs - 1):
        for j in range(i+1, nObs):
            dist = euclideandistance(input[i, :], input[j,:])
            if dist > RMax:
                RMax = dist
    return RMax/2
