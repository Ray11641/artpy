"""
    SalientART PY: A Python library of Salient ART. This file is part of Neurol-
    ogy phenotyping.
"""

import numpy as np

def ComplementCoding(iNput,dim=0):

    """
    :param iNput: iNput data
    :param dim: dimension along with normalisation should be done
    """

    if isinstance(iNput,list):
        nSamples = len(iNput)
        tiNput = np.array(iNput)
        dim_max = np.max(tiNput, axis = dim)
        dim_min = np.min(tiNput, axis = dim)

        normalised = (tiNput-dim_min)/(dim_max - dim_min)
        normalised = np.concatenate((normalised, 1- normalised), axis=1)
        CCiNput = []
        for itr in range(nSamples):
            CCiNput.append(normalised[itr,:])

    elif isinstance(iNput, np.ndarray):
        dim_max = np.max(iNput, axis = dim)
        dim_min = np.min(iNput, axis = dim)

        if dim == 0:
            normalised = (iNput-dim_min)/(dim_max - dim_min)
        if dim ==1:
            normalised = ((iNput.T - dim_min)/(dim_max- dim_min)).T

        CCiNput = np.concatenate((normalised,1-normalised), axis = 1-dim)
    else:
        raise TypeError("Not either a list or an np.ndarray")

    return CCiNput
