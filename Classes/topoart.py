"""
    ARTPY: A Python library of ART Nerual Networks.
"""

import os
import numpy as np

class TopoART:
    """
       Reference: Tscherepanow, M., 2010. TopoART: A topology learning hierarchical
       ART network. In Artificial Neural Networks–ICANN 2010: 20th International
       Conference, Thessaloniki, Greece, September 15-18, 2010, Proceedings, Part
       III 20 (pp. 157-167). Springer Berlin Heidelberg.
    """

    def __init__( self, vigilance, alpha, beta1, beta2, phi, tau, nlevels):
        """
        :Param vigilance: vigilance value for training the ART model
        :Param alpha: The parameter for the choice function evaluation
        :Param beta1: Learning rate for training the first winner
        :Param beta2: Learning rate for training the second winner
        :Param phi: The minimum number of samples to be summarised to be a permanent
                    prototype
        :Param tau: The number of time steps for pruning temporary prototypes
        """
        self.vigilance = vigilance
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.phi = phi
        self.prototypes = {}
        self.labels_ = {}
        for levels in range(nlevels):
            self.prototypes[f"levels"] = []
            self.labels_[f"levels"] = []

    def choice(self,input):
        T = []
        for prototype in self.prototypes:
            choice = np.sum(np.minimum(prototype,input))/(self.alpha + np.sum(prototype))
            T.append(choice)
        return T

    def match(self,input):
        M = []
        for prototype in self.prototypes:
            match = np.sum(np.minimum(prototype, input))/np.sum(input)
            M.append(match)
        return M

    def learn(self,input):
        """
        :Param input: the input vector to be fed the ART model
        """


    def fit(self,data):
        """
        :Param data: the input data for the ART model
        """
