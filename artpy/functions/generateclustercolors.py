"""
    SalientART PY: A Python library of Salient ART.
    This code is sourced from ARTPY: A Python library
    of Adaptive Resonance Theory based learning models.

    This file provides rgb2hex and generateclustcolor functions.
"""

from typing import Dict, List, Tuple, IO
import numpy as np
import matplotlib as mplib

__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2023"
__credits__ = ["Leonardo Enzo Brito Da Silva", "Donald Wunsch"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Raghu Yelugam"
__email__ = "ry222@mst.edu"
__status__ = "Development"
__date__ = "2023.04.13"


def rgb2hex(r: int,
            g: int,
            b: int) -> str:
    """
    :param r: red value in range of 0-255
    :param g: green value in range of 0-255
    :param b: blue value in range of 0-255
    """
    try:
        if r<0 or r>255:
            raise ValueError('color channels should be or range [0, 255] or [0, 1]')
        if g<0 or g>255:
            raise ValueError('color channels should be or range [0, 255] or [0, 1]')
        if b<0 or b>255:
            raise ValueError('color channels should be or range [0, 255] or [0, 1]')
        check1: bool = isinstance(r, float) and \
                       isinstance(g, float) and \
                       isinstance(b, float) 
        check2: bool = isinstance(r, int) and \
                       isinstance(g, int) and \
                       isinstance(b, int) 
        if not check1^check2:
            raise ValueError('color channels should be either all floats or all ints')
            
    except Exception as e:
        print(f"following error occured in rgb2hex: {e}")
        
    else:
        if isinstance(r, float) and r<=1.0:
            r = int(255*r)
        if isinstance(g, float) and g<=1.0:
            g = int(255*g)
        if isinstance(b, float) and b<=1.0:
            b = int(255*b)
        return "#{:02x}{:02x}{:02x}".format(r,g,b)


def generateclustcolors(ncolourcodes_: int,
                        colormap_: str = "viridis") -> List[str]:
    """
    This function generates ncolourcodes_ sampled from matplotlib colormap
    :param colormap_: matplotlib colour map to be used for generating the colours
    :param ncolourcodes_: number of colour codes needed
    """
    
    clust_colors = []
    colourfunc = mplib.pyplot.get_cmap(colormap_)
    for itr in range(ncolourcodes_):
        obj = colourfunc(itr/(ncolourcodes_ - 1))
        clust_colors.append(rgb2hex(obj[0],
                                    obj[1],
                                    obj[2]))
    return clust_colors