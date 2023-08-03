"""
    ARTPY: A Python library of Adaptive Resonance Theory based learning
     models.

     This file provides distance functions.
"""

import typing
from typing import List, Union, Tuple
import numpy as np

def euclideandistance(a: Union[np.ndarray, list],
             b: Union[np.ndarray, list]) -> float:
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    dist = np.sqrt(np.sum((a - b)**2))
    return dist
