"""
    ARTPY: A Python library of Adaptive Resonance Theory based learning
     models.

     This file provides TopoART class.
"""

import os
from typing import Dict, List, Tuple, IO
from operator import itemgetter
import numpy as np
import networkx as nx
from pyvis.network import Network
from .. functions import *
import ipdb

__author__ = "Raghu Yelugam"
__copyright__ = "Copyright 2023"
__credits__ = ["Leonardo Enzo Brito Da Silva", "Donald Wunsch"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Raghu Yelugam"
__email__ = "ry222@mst.edu"
__status__ = "Development"
__date__ = "2023.04.13"


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
                 tau_: int) -> None:
        """
        :param vigilance_: vigilance value for training the ART model
        :param alpha_: The parameter for the choice function evaluation
        :param beta1_: Learning rate for training the first winner
        :param beta2_: Learning rate for training the second winner
        :param phi_: The minimum number of samples to be summarised to 
                be a permanent prototype
        :param tau_: The number of time steps for pruning temporary prototypes
        """
        self.vigilance_: float = vigilance_
        self.alpha_: float = alpha_
        self.beta1_: float = beta1_
        self.beta2_: float = beta2_
        self.phi_: float = phi_
        self.cycle_: int = 0
        self.tau_: int = tau_
        self.prototypes_: Dict[List[np.ndarray], List[int], List[str]] = {"weights": [],
                                                                         "counter": [],
                                                                         "tag": []}       
        self.__labels_: List[str] = []
        self.edges_: List[Tuple[str, str]] = []
        self.topoClusters_: List[List[str]] = []
        self.labels_: List[int] = []
        self.__addedTags: List[str] = []
        
    def choice(self,
               input: np.ndarray) -> List[float]:
        """
        :param input: current input
        """
        T: List[float] = []
        for prototype in self.prototypes_["weights"]:
            choice = np.sum(np.minimum(prototype, input)) / \
                    (self.alpha_ + np.sum(prototype))
            T.append(choice)
        return T

    def match(self,
              input: np.ndarray) -> List[float]:
        M: List[float] = []
        for prototype in self.prototypes_["weights"]:
            match = np.sum(np.minimum(prototype, input))/np.sum(input)
            M.append(match)
        return M
    
    def prune(self) -> None:
        """
        prune the prototypes with count less than self.tau_
        replace label for samples summarised by pruned prototypes
        with "d"
        """
        counts = np.array(self.prototypes_["counter"])
        pruneLocations = np.where(counts<self.phi_)[0]
        pruneLocations[::-1].sort()
        tags = []
        for index in pruneLocations:
            del self.prototypes_["weights"][index]
            del self.prototypes_["counter"][index]
            tags.append(self.prototypes_["tag"].pop(index))
        itr = 0
        while itr < len(self.edges_):
            if self.edges_[itr][0] in tags \
             or self.edges_[itr][1] in tags:
                self.edges_.remove(self.edges_[itr])
                itr -= 1
            itr += 1
        for (loc, x) in enumerate(self.__labels_):
            if x in tags:
                self.__labels_[loc] = "d"

    def linkedges(self) -> None:
        """
        This function identifies the topological clusters in the data.
        """
        for edge in self.edges_:
            if edge[0] not in self.__addedTags:
                self.__addedTags.append(edge[0])
            if edge[1] not in self.__addedTags:
                self.__addedTags.append(edge[1])
                
            if len(self.topoClusters_) == 0:
                tList = []
                tList.append(edge[0])
                tList.append(edge[1])
                self.topoClusters_.append(tList)
            else:
                ntC = len(self.topoClusters_)
                tag0Loc = None
                tag1Loc = None
                
                for index in range(ntC):
                    if edge[0] in self.topoClusters_[index]:
                        tag0Loc = index
                    if edge[1] in self.topoClusters_[index]:
                        tag1Loc = index
                if tag0Loc == None and tag1Loc == None:
                    self.topoClusters_.append([edge[0],edge[1]])
                elif tag0Loc == None:
                    self.topoClusters_[tag1Loc].append(edge[0])
                elif tag1Loc == None:
                    self.topoClusters_[tag0Loc].append(edge[1])
                else:
                    if tag0Loc != tag1Loc:
                        self.topoClusters_[tag0Loc].extend(self.topoClusters_[tag1Loc])
                        del self.topoClusters_[tag1Loc]

        for tag in self.prototypes_["tag"]:
            if tag not in self.__addedTags:
                self.topoClusters_.append([tag])
                self.__addedTags.append(tag)

    def classify(self,
                input: np.ndarray) -> List[int]:
        """
        :param input: Input data to be classified
        """
        input = ComplementCoding(input)
        labels: List[int] =  []
        for val in input:
            T: List[float] = []
            for prototype in self.prototypes_["weights"]:
                choice = 1 - (np.sum(np.minimum(prototype, val) - prototype) / \
                        (np.sum(val)))
                T.append(choice)
            location = choice.index(max(choice))

            tag = self.prototypes_["tag"][location]
            for itr in range(len(self.topoClusters_)):
                if tag in self.topoClusters_[itr]:
                    T.append(itr)

    def label(self) -> None:
        self.labels_ = []
        for tag in self.__labels_:
            if tag == "d":
                self.labels_.append(-1)
            else:
                for itr in range(len(self.topoClusters_)):
                    if tag in self.topoClusters_[itr]:
                        self.labels_.append(itr)

    def getgraph(self) -> IO:
        nclusters = len(self.topoClusters_)
        nodes = self.prototypes_["tag"]
        colours = generateclustcolors(nclusters)
        node_colour = []
        for tag in self.prototypes_["tag"]:
            for loc, cluster in enumerate(self.topoClusters_):
                if tag in cluster:
                    node_colour.append(colours[loc])
                    break
        G = Network()
        G.add_nodes(nodes,
                    color = node_colour)
        G.add_edges(self.edges_)
        return G
            
    def getlocallabels(self,) -> list:
        return self.__labels_

    def learn(self,
              input: np.ndarray) -> None:
        """
        :param input: the input vector to be fed the ART model
        """
        level: int = 0
        self.cycle_ += 1
        if len(self.prototypes_["weights"]) == 0:
            self.prototypes_["weights"].append(input)
            self.prototypes_["counter"].append(1)
            self.prototypes_["tag"].append(f'p{self.cycle_}')
            self.__labels_.append(f'p{self.cycle_}')

        else:
            T = self.choice(input)
            M = self.match(input)
            l = len(self.__labels_)
            while not all(val < 0 for val in T):
                IFW: int = T.index(max(T))
                if M[IFW] >= self.vigilance_:
                    self.prototypes_["weights"][IFW] = \
                        (1-self.beta1_)*self.prototypes_["weights"][IFW] \
                        + self.beta1_*np.minimum(input,self.prototypes_["weights"][IFW])
                    self.prototypes_["counter"][IFW] += 1
                    tagFW = self.prototypes_["tag"][IFW]
                    self.__labels_.append(tagFW)
                    T[IFW] = -1.0

                    while not all(val < 0 for val in T):
                        ISW: int = T.index(max(T))
                        if M[ISW] >= self.vigilance_:
                            self.prototypes_["weights"][ISW] = \
                                (1-self.beta2_)*self.prototypes_["weights"][ISW] \
                                + self.beta2_*np.minimum(input,self.prototypes_["weights"][ISW])
                            tagSW = self.prototypes_["tag"][ISW]
                            if (tagFW, tagSW) not in self.edges_:
                                self.edges_.append((tagFW, tagSW))
                            break
                        else:
                            T[ISW] = -1.0
                    break
                else:
                    T[IFW] = -1.0

            if all(val < 0 for val in T) and \
             l == len(self.__labels_):
                self.prototypes_["weights"].append(input)
                self.prototypes_["counter"].append(1)
                self.prototypes_["tag"].append(f'p{self.cycle_}')
                self.__labels_.append(f'p{self.cycle_}')

        if self.cycle_%self.tau_ == 0:
            self.prune()
            self.linkedges()
            self.label()

    def fit(self,
            data: np.ndarray,
            verbose: bool = False) -> None:
        """
        :param data: the input data for the ART model
        """
        data = ComplementCoding(data)
        temp = 0
        for val in data:
            temp += 1
            if verbose:
                print(f"Presenting observation #{temp}")
            self.learn(val)
        self.prune()
        self.linkedges()
        self.label()
        if verbose:
            print("Done learning")

