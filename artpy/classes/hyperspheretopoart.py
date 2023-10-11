"""
    ARTPY: A Python library of Adaptive Resonance Theory based learning
     models.

    This file provides HypersphereART class.
"""

import os
from typing import Dict, List, Tuple, IO
#from operator import itemgetter
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
__status__ = "Release"
__date__ = "2023.04.13"


class HypersphereTopoART:
    """
    Tscherepanow, Marko. "Incremental On-line Clustering with a
    Topology-Learning Hierarchical ART Neural Network Using 
    Hyperspherical Categories." In ICDM (Poster and Industry
    Proceedings), pp. 22-34. 2012.
    """

    def __init__(self,
                 vigilance_: float,
                 alpha_: float,
                 beta1_: float,
                 beta2_: float,
                 radialextend_: float,
                 rmax_: float,
                 phi_: int,
                 tau_: int) -> None:
        """
        :param vigilance_: vigilance value for training the ART model
        :param alpha_: The parameter for the choice function evaluation
        :param beta1_: Learning rate for training the first winner
        :param beta2_: Learning rate for training the second winner
        :param radialextend_: The radial extension parameter, should be a value
            between [rmax_, inf)
        :param rmax_: The maximum radius of the presented data
        :param phi_: The minimum number of samples to be summarised to
                be a permanent prototype
        :param tau_: The number of time steps for pruning temporary prototypes
        radialextendu_ refers to uncommitted nodes radialextend
        """
        if radialextend_ < rmax_:
            error = f"expected radialextend_ ({radialextend_}) >= rmax_ ({rmax_})"
            raise Exception(error)

        self.vigilance_: float = vigilance_
        self.alpha_: float = alpha_
        self.beta1_: float = beta1_
        self.beta2_: float = beta2_
        self.phi_: float = phi_
        self.cycle_: int = 0
        self.tau_: int = tau_
        self.prototypes_: Dict[List[List[np.ndarray, float]], List[int], List[str]] = {}
        self.prototypes_["weights"] = []
        self.prototypes_["counter"] = []
        self.prototypes_["tag"] = []
        self.__labels_: List[str] = []
        self.edges_: List[Tuple[str, str]] = []
        self.topoClusters_: List[List[str]] = []
        self.labels_: List[int] = []
        self.__addedTags: List[str] = []
        self.radialextend_ = radialextend_
        self.rmax_ = rmax_
        self.radialextendu_ = 2*self.radialextend_

    def __repr__(self) -> str:
        #Change this
        v = self.vigilance_
        a = self.alpha_
        b = self.beta_
        re = self.radialextend_
        rm = self.rmax_
        return f"HypersphereTopoART(vigilance = {v}, alpha = {a}, beta = {b}, radialextend = {re}, rmax = {rm})"

    def choice(self,
               input: np.ndarray) -> List[float]:
        """
        :param input: current input
        """
        T: List[float] = []
        for prototype in self.prototypes_["weights"]:
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
        M = max(self.prototypes_["weights"][index][1],
            euclideandistance(self.prototypes_["weights"][index][0], input))
        return 1 - (M/self.radialextend_)

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
        #This has to change
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
        self.cycle_ += 1
        if len(self.prototypes_["weights"]) == 0:
            self.prototypes_["weights"].append([input, 0])
            self.prototypes_["counter"].append(1)
            self.prototypes_["tag"].append(f'p{self.cycle_}')
            self.__labels_.append(f'p{self.cycle_}')

        else:
            T = self.choice(input)
            while not all(val < 0 for val in T):
                IFW: int = T.index(max(T))
                print(f"Index = {IFW}")
                length = len(self.prototypes_["weights"])
                print(f"#prototypes = {length}")
                if IFW == len(self.prototypes_["weights"]):
                    self.prototypes_["weights"].append([input, 0])
                    self.prototypes_["counter"].append(1)
                    self.prototypes_["tag"].append(f'p{self.cycle_}')
                    self.__labels_.append(f'p{self.cycle_}')
                    break
                else:
                    M: float = self.match(input, IFW)
                    if M >= self.vigilance_:
                        dist = euclideandistance(input, self.prototypes_["weights"][IFW][0])
                        a: float = 1 - min(self.prototypes_["weights"][IFW][1], dist)/dist
                        b: float = input - self.prototypes_["weights"][IFW][0]
                        self.prototypes_["weights"][IFW][0] += self.beta1_*a*b/2
                        a = max(self.prototypes_["weights"][IFW][1], dist)
                        b = self.prototypes_["weights"][IFW][1]
                        self.prototypes_["weights"][IFW][1] += self.beta1_*(a - b)/2
                        self.prototypes_["counter"][IFW] += 1
                        tagFW = self.prototypes_["tag"][IFW]
                        self.__labels_.append(tagFW)
                        T[IFW] = -1.0
                        while not all(val < 0 for val in T):
                            ISW: int = T.index(max(T))
                            if ISW == len(self.prototypes_["weights"]):
                                break
                            M: float = self.match(input, ISW)
                            if M >= self.vigilance_:
                                dist = euclideandistance(input, self.prototypes_["weights"][IFW][0])
                                a: float = 1 - min(self.prototypes_["weights"][IFW][1], dist)/dist
                                b: float = input - self.prototypes_["weights"][IFW][0]
                                self.prototypes_["weights"][IFW][0] += self.beta2_*a*b/2
                                a = max(self.prototypes_["weights"][IFW][1], dist)
                                b = self.prototypes_["weights"][IFW][1]
                                self.prototypes_["weights"][IFW][1] += self.beta2_*(a - b)/2
                                tagSW = self.prototypes_["tag"][ISW]
                                if (tagFW, tagSW) not in self.edges_:
                                    self.edges_.append((tagFW, tagSW))
                                break
                            else:
                                T[ISW] = -1.0
                        break
                    else:
                        T[IFW] = -1.0
        
        if self.cycle_%self.tau_ == 0:
            self.prune()
            self.linkedges()
            self.label()

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

        self.prune()
        self.linkedges()
        self.label()
        if verbose:
            print("Done learning")
