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
        self.tau_: int = tau_
        self.prototypes: dict = {}
        self.labels_: dict = {}
        self.edges_: Dict[List[Tuple[int, int]]] = {}
        for levels in range(nlevels):
            self.vigilance_.append(vigilance_)
            vigilance_ = (1+vigilance_)/2
            self.prototypes[f"{levels}"]: Dict[list, list, list] = {"weights": [],
                                                                    "counter": [],
                                                                    "tag": []}
            self.labels_[f"{levels}"]: List[int] = []
            self.edges_[f"{levels}"]: List[Tuple[int, int]] = []

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
    
    def prune(self,
              level: int) -> None:
        """
        prune the prototypes with count less than self.tau
        :param level: the level of the hierarchy to be pruned
        """
        counts = np.array(self.prototypes[f"level"]["counter"])
        pruneLocations = np.where(counts<self.phi_)[0]
        pruneLocations[::-1].sort()
        for index in pruneLocations:
            self.prototypes[f"level"]["weights"].pop(index)
            self.prototypes[f"level"]["counter"].pop(index)
            tag = self.prototypes[f"level"]["tag"][index]
            for edge in self.edges_[f"level"]:
                if tag in edge:
                    edges.remove(edge)
            for (loc, x) in enumearate(self.labels_[f"level"]):
                if x == tag:
                    self.labels_[f"level"][loc] = "d"
            self.prototypes[f"level"]["tag"].pop(index)
                

    def learn(self,
              input: np.ndarray) -> None:
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
                self.prototypes[f"level"]["tag"].append(f'p{self.cycle_}')
                self.labels_[f"level"].append(f'p{self.cycle_}')
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
                        tagFW = self.prototypes[f"level"]["tag"][IFW]
                        self.labels_[level].append(tagFW)
                        T[IFW] = -1.0
                        if self.prototypes[f"level"]["counter"][IFW] < self.phi_:
                            PassToLevel = False
                        
                        while not all(val < 0 for val in T):
                            ISW: int = T.index(max(T))
                            if M[ISW] >= self.vigilance_[level]:
                                self.prototypes[f"level"]["weights"][ISW] = \
                                    (1-self.beta2_)*self.prototypes[f"level"]["weights"][ISW] \
                                    + self.beta2_*np.minimum(input,self.prototypes[f"level"]["weights"][ISW])
                                tagSW = self.prototypes[f"level"]["tag"][SFW]
                                self.edges_[f"level"].append((tagFW, tagSW))
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
                    self.labels_[f"level"].append(f'p{self.cycle_}')
                    self.prototypes[f"level"]["tag"].append(f'p{self.cycle_}')
                    PassToLevel = False

            if not PassToLevel:
                for itr in range(level, self.nlevels):
                    self.labels_[[f"itr"]].append(-1)

        if self.cycle_%self.tau_ == 0:
            for level in range(self.nlevels):
                self.prune(level)