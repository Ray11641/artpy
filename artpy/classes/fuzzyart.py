"""
    ARTPY: A Python library of Adaptive Resonance Theory based learning
     models.

     This file provides FuzzyART class.
"""
__version__ = "0.1"
__author__ = "Raghu Yelugam"

import os
import numpy as np
from typing import Dict, List, Tuple


class FuzzyART:
    """
       Reference: Carpenter, G.A., Grossberg, S. and Rosen, D.B., 1991.
       Fuzzy ART: Fast stable learning and categorization of analog
       patterns by an adaptive resonance system. Neural networks, 4(6),
       pp.759-771.

    """

    def __init__(self,
                 vigilance_: float,
                 alpha_: float,
                 beta_: float) -> None:
        """
        :param vigilance_: vigilance value for training the ART model
        :param alpha_: The parameter for the choice function evaluation
        :param beta_: Learning rate for training the Fuzzy ART model
        """
        self.vigilance_ = vigilance_
        self.alpha_ = alpha_
        self.beta_ = beta_
        self.prototypes: List[np.ndarray] = []
        self.labels_: List[int] = []

    def __repr__(self) -> str:
        v = self.vigilance_
        b = self.beta_
        a = self.alpha_
        return f"FuzzyART(vigilance ='{v}', alpha = '{a}', beta = '{b}')"

    def choice(self,
               input: np.ndarray) -> List[float]:
        """
        :param input: current input
        """
        T: List[float] = []
        for prototype in self.prototypes:
            choice = np.sum(np.minimum(prototype, input))
            choice /= self.alpha_ + np.sum(prototype)
            T.append(choice)
        return T

    def match(self,
              input: np.ndarray) -> List[float]:
        """
        :param input: current input
        """

        M: List[float] = []
        for prototype in self.prototypes:
            if np.sum(prototype) == 0 and np.sum(input) == 0:
                match = 1
            else:
                match = np.sum(np.minimum(prototype, input))/np.sum(input)
            M.append(match)
        return M

    def learn(self,
              input: np.ndarray) -> None:
        """
        :param input: the input vector to be fed the ART model
        """
        if len(self.prototypes) == 0:
            self.prototypes.append(input)
            self.labels_.append(0)
        else:

            T = self.choice(input)
            M = self.match(input)
            while not all(val < 0 for val in T):
                I: int = T.index(max(T))
                if M[I] >= self.vigilance_:
                    self.prototypes[I] = (1 - self.beta_)*self.prototypes[I] \
                            + self.beta_*np.minimum(input, self.prototypes[I])
                    self.labels_.append(I)
                    break
                else:
                    T[I] = -1.0

            if all(val < 0 for val in T):
                self.prototypes.append(input)
                self.labels_.append(len(self.prototypes)-1)

    def fit(self,
            data: np.ndarray) -> None:
        """
        :param data: the input data for the ART model
        """
        temp = 0
        for val in data:
            temp += 1
            print(f"Presenting observation #{temp}")
            self.learn(val)
        print("Done learning")
