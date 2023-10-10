"""
    ARTPY: A Python library of Adaptive Resonance Theory based learning
     models.

    This file provides HypersphereART class.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Union
from ..functions import *

__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2023"
__credits__ = ["Leonardo Enzo Brito Da Silva", "Donald Wunsch"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Raghu Yelugam"
__email__ = "ry222@mst.edu"
__status__ = "Development"
__date__ = "2023.04.13"


class HypersphereART:
    """
       Reference: Anagnostopoulos, G.C. and Georgiopulos, M., 2000,
       July. Hypersphere ART and ARTMAP for unsupervised and
       supervised, incremental learning. In Proceedings of the
       IEEE-INNS-ENNS International Joint Conference on Neural Networks.
       IJCNN 2000. Neural Computing: New Challenges and Perspectives
       for the New Millennium (Vol. 6, pp. 59-64). IEEE.
    """

    def __init__(self,
                 vigilance_: float,
                 alpha_: float,
                 beta_: float,
                 radialextend_: float,
                 rmax_: float) -> None:
        """
        :param vigilance_: vigilance value for training the ART model
        :param alpha_: The parameter for the choice function evaluation
        :param beta_: Learning rate for training the ART model
        :param radialextend_: The radial extension parameter, should be a value
            between [rmax_, inf)
        :param rmax_: The maximum radius of the presented data
        radialextendu_ refers to uncommitted nodes radialextend
        """
        self.vigilance_ = vigilance_
        self.alpha_ = alpha_
        self.beta_ = beta_
        self.prototypes_: List[List[np.ndarray, float]] = []
        self.labels_: List[int] = []
        if radialextend_ < rmax_:
            error = f"expected radialextend_ ({radialextend_}) >= rmax_ ({rmax_})"
            raise Exception(error)
        self.radialextend_ = radialextend_
        self.rmax_ = rmax_
        self.radialextendu_ = 2*self.radialextend_

    def __repr__(self) -> str:
        v = self.vigilance_
        a = self.alpha_
        b = self.beta_
        re = self.radialextend_
        rm = self.rmax_
        return f"HypershpereART(vigilance = {v}, alpha = {a}, beta = {b}, radialextend = {re}, rmax = {rm})"

    def choice(self,
               input: np.ndarray) -> List[float]:
        """
        :param input: current input
        """
        T: List[float] = []
        for prototype in self.prototypes_:
            choice = self.radialextend_ - max(prototype[1],
                                              euclideandistance(prototype[0], input))
            choice /= self.radialextend_ - prototype[1] + self.alpha_
            T.append(choice)
        T.append(self.radialextend_/(self.radialextendu_ + self.alpha_))
        return T

    def match(self,
              input: np.ndarray,
              index: int) -> float:
        """
        :param input: current input
        """
        M = max(self.prototypes_[index][1],
            euclideandistance(self.prototypes_[index][0], input))
        return 1 - (M/self.radialextend_)

    def learn(self,
              input: np.ndarray) -> None:
        """
        :param input: the input vector to be fed the ART model
        """
        if len(self.prototypes_) == 0:
            self.prototypes_.append([input, 0])
            self.labels_.append(0)
        else:
            T = self.choice(input)
            while not all(val < 0 for val in T):
                I: int = T.index(max(T))
                if I == len(self.prototypes_):
                    self.prototypes_.append([input, 0])
                    self.labels_.append(len(self.prototypes_) - 1)
                    break
                else:
                    M: float = self.match(input, I)
                    if M >= self.vigilance_:
                        dist = euclideandistance(input, self.prototypes_[I][0])
                        a: float = 1 - min(self.prototypes_[I][1], dist)/dist
                        b: float = input - self.prototypes_[I][0]
                        self.prototypes_[I][0] += self.beta_*a*b/2
                        a = max(self.prototypes_[I][1], dist)
                        b = self.prototypes_[I][1]
                        self.prototypes_[I][1] += self.beta_*(a - b)/2
                        self.labels_.append(I)
                        break
                    else:
                        T[I] = -1.0

    def fit(self,
            data: np.ndarray,
            verbose: bool = False) -> None:
        """
        :param data: the input data for the ART model
        :param verbose: to print verbose
        """
        if verbose:
            temp = 0
        for val in data:
            if verbose:
                temp += 1
                print(f"Presenting observation #{temp}")
            self.learn(val)

        if verbose:
            print("Done learning")