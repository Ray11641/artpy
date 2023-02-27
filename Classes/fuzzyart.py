"""
    SalientART PY: A Python library of Salient ART.
"""

import os
import numpy as np

class FuzzyART:
    """
       Reference: Carpenter, G.A., Grossberg, S. and Rosen, D.B., 1991.
       Fuzzy ART: Fast stable learning and categorization of analog
       patterns by an adaptive resonance system. Neural networks, 4(6),
       pp.759-771.

    """

    def __init__(self,vigilance,alpha, beta):
        """
        :Param vigilance: vigilance value for training the ART model
        :Param alpha: The parameter for the choice function evaluation
        :Param beta: Learning rate for training the Fuzzy ART model
        """
        self.vigilance = vigilance
        self.alpha = alpha
        self.beta = beta
        self.prototypes = []
        self.labels_ = []

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
        if len(self.prototypes) == 0:
            self.prototypes.append(input)
            self.labels_.append(0)
        else:

            T = self.choice(input)
            M = self.match(input)
            while not all(val<0 for val in T):
                #print("In While Loop")
                I = T.index(max(T))
                if M[I] >= self.vigilance:
                    self.prototypes[I] =  (1 -self.beta)*self.prototypes[I] + self.beta*np.minimum(input,self.prototypes[I])
                    self.labels_.append(I)
                    break
                else:
                    T[I] = -1.0

            if all(val<0 for val in T):
                self.prototypes.append(input)
                self.labels_.append(len(self.prototypes)+1)

    def fit(self,data):
        """
        :Param data: the input data for the ART model
        """
        temp = 0
        for val in data:
            temp+=1
            print(f"Input number #{temp}")
            self.learn(val)
        print("Done self.learn")
