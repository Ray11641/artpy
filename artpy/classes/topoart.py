"""
    ARTPY: A Python library of Adaptive Resonance Theory based learning
     models.

     This file provides TopoART class.
"""

import os
import numpy as np
from typing import Dict, List, Tuple


class TopoART:
    """
       Reference: Tscherepanow, M., 2010. TopoART: A topology learning hierarc
       -hical ART network. In Artificial Neural Networks–ICANN 2010: 20th Inte
       -rnational Conference, Thessaloniki, Greece, September 15-18, 2010, 
       Proceedings, Part III 20 (pp. 157-167). Springer Berlin Heidelberg.
    """

    def __init__(self,
                 vigilance_: float,
                 alpha_: float,
                 beta1_: float,
                 beta2_: float,
                 phi_: int,
                 tau_: int,
                 nlevels: int = 1) -> None:
        """
        :param vigilance_: vigilance value for training the ART model
        :param alpha_: The parameter for the choice function evaluation
        :param beta1_: Learning rate for training the first winner
        :param beta2_: Learning rate for training the second winner
        :param phi_: The minimum number of samples to be summarised to 
                be a permanent prototype
        :param tau_: The number of time steps for pruning temporary prototypes
        """
        self.vigilance_: List[int] = []
        self.alpha_ = alpha_
        self.beta1_ = beta1_
        self.beta2_ = beta2_
        self.phi_ = phi_
        self.nlevels: int = nlevels
        self.cycle_: int = 0
        self.prototypes: dict = {}
        self.labels_: dict = {}
        self.edges: List[List[Tuple[int, int]]] = {}
        for levels in range(nlevels):
            self.vigilance_.append(vigilance_)
            vigilance_ = (1+vigilance_)/2
            self.prototypes[f"{levels}"]: Dict[list, list] = {"weights": [],
                                                              "counter": []}
            self.labels_[f"{levels}"]: List[int] = []


    def choice(self,
               input: np.ndarray,
               level: int = 0) -> List[float]:
        """
        :param input: current input
        :param level: the heirarchy level, default set to lowest, 0
        """
        T: List[float] = []
        for prototype in self.prototypes[f"{level}"]:
            choice = np.sum(np.minimum(prototype, input)) / \
                    (self.alpha + np.sum(prototype))
            T.append(choice)
        return T

    def match(self,
              input: np.ndarray,
              level: int = 0) -> List[float]:
        M: List[float] = []
        for prototype in self.prototypes[f"{level}"]:
            match = np.sum(np.minimum(prototype, input))/np.sum(input)
            M.append(match)
        return M

    def learn(self, input):
        """
        :param input: the input vector to be fed the ART model
        """
        PassToLevel: bool = True
        level: int = 0
        self.cycle_ += 1
        while PassToLevel:
            if len(self.prototypes[f"level"]) == 0:
                self.prototypes[f"level"]["weights"].append(input)
                self.prototypes[f"level"]["counter"].append(1)
                self.labels_[f"level"].append(0)
                PassToLevel = False

            else:
                T = self.choice(input, level)
                M = self.match(input, level)
                while not all(val < 0 for val in T):
                    IFW: int = T.index(max(T))
                    if M[IFW] >= self.vigilance_[level]:
                        self.prototypes[f"level"]["weights"][IFW] = \
                            (1-self.beta1_)*self.prototypes[f"level"]["weights"][IFW] \
                            + self.beta1_*np.minimum(input,self.prototypes[f"level"]["weights"][IFW])
                        self.prototypes[f"level"]["counter"][IFW] += 1
                        self.labels_[level].append(IFW)
                        T[IFW] = -1.0
                        if self.prototypes[f"level"]["counter"][IFW] < self.phi_:
                            PassToLevel = False
                        
                        while not all(val < 0 for val in T):
                            ISW: int = T.index(max(T))
                            if M[ISW] >= self.vigilance_[level]:
                                self.prototypes[f"level"]["weights"][ISW] = \
                                    (1-self.beta2_)*self.prototypes[f"level"]["weights"][ISW] \
                                    + self.beta2_*np.minimum(input,self.prototypes[f"level"]["weights"][ISW])
                                #Add edges
                                break
                            else:
                                T[ISW] = -1.0

                        break

                    else:
                        T[IFW] = -1.0

                if all(val < 0 for val in T):
                    self.prototypes[f"level"]["weights"].append(input)
                    self.prototypes[f"level"]["counter"].append(1)
                    newindex: int = len(self.prototypes[f"level"]["weights"])
                    self.labels_[f"level"].append(newindex-1)
                    PassToLevel = False

            if not PassToLevel:
                for itr in range(level, self.nlevels):
                    self.labels_[[f"itr"]].append(-1)


        if self.cycle_%self.tau_ == 0:
            #prune the edges
            for level in range(self.nlevels):
                self.prune(level)