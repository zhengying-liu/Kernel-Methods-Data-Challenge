"""
Fisher Vector.
See https://hal.inria.fr/hal-00779493v3/document page 14 algorithm 1 for more details.
"""

import numpy as np

from gamma import gamma
from gmm import Gmm

class FisherVector:
    """
    nclasses: number of classes (assumed between 0 and nclasses - 1)
    dim: dimension of the space in which the data lives
    pi: proportion of different classes
    mu: centers of different classes
    sigma: diagonal values of the covariance matrix, represented by a 2d numpy array
    (Note that our sigma is sigma^2 in the paper)
    stat0: statistics 0, 1d array of length nclassses
    stat1: statistics 1, 2d array of shape (nclasses, dim)
    stat2: statistics 2, 2d array of shape (nclasses, dim)
    fv: Fisher vector, 1d array of length nclasses * (2 * dim + 1)
    """

    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.dim = None
        self.pi = None
        self.mu = None
        self.sigma = None
        self.stat0 = None
        self.stat1 = None
        self.stat2 = None
        self.fv = None
        
    def _gmm_fit(self, data, niter=10):
        self.dim = len(data[0])
        gmm = Gmm(self.nclasses)
        gmm.fit(data, niter)
        self.pi = gmm.pi
        self.mu = gmm.mu
        self.sigma = gmm.sigma
        
    def _compute_statistics(self, data):
        ndata = len(data)
        gamma_K_N = np.empty((self.nclasses, ndata))
        sigma_matrix = np.zeros((self.nclasses, self.dim, self.dim))
        for k in range(self.nclasses):
            sigma_matrix[k] = np.diag(self.sigma[k])
        for t in range(ndata):
            gamma_K_N[:,t] = gamma(data[t], self.pi, self.mu, sigma_matrix)[1]
            
        self.stat0 = np.zeros(self.nclasses)
        self.stat1 = np.zeros((self.nclasses, self.dim))
        self.stat2 = np.zeros((self.nclasses, self.dim))
        for t in range(ndata):
            for k in range(self.nclasses):
                self.stat0[k] += gamma_K_N[k, t]
                self.stat1[k] += gamma_K_N[k, t] * data[t]
                self.stat2[k] += gamma_K_N[k, t] * data[t] * data[t]
    
    def _compute_signature(self, data):
        ndata = len(data)
        self.fv = np.zeros(self.nclasses * (2 * self.dim + 1))
        for k in range(self.nclasses):
            self.fv[k] = (self.stat0[k] - ndata * self.pi[k]) / np.sqrt(self.pi[k])
        
        offset = self.nclasses
        for k in range(self.nclasses):
            signature_temp = (self.stat1[k, :] - self.mu[k, :] * self.stat0[k])/ (np.sqrt(self.pi[k] * self.sigma[k, :]))
            for d in range(self.dim):
                self.fv[offset + d] = signature_temp[0, d]
            offset += self.dim
        
        for k in range(self.nclasses):
            signature_temp = (self.stat2[k, :] * self.stat2[k, :] - 2 * self.mu[k, :][:, 0] * self.stat1[k, :] + (self.mu[k, :][:, 0] * self.mu[k, :] - self.sigma[k, :]) * self.stat0[k]) / (np.sqrt(2 * self.pi[k]) * self.sigma[k, :])
            for d in range(self.dim):
                self.fv[offset + d] = signature_temp[0, d]
            offset += self.dim
        
    def _normalize(self):
        for i in range(len(self.fv)):
            self.fv[i] = np.sign(self.fv[i]) * np.sqrt(np.abs(self.fv[i]))
        self.fv = self.fv / np.linalg.norm(self.fv, ord=2)
        
    def predict(self, data):
        self._gmm_fit(data)
        self._compute_statistics(data)
        self._compute_signature(data)
        self._normalize()
        return self.fv
        
if __name__ == '__main__':
    my_data = np.array([[1, 2, 3], [4, 5, 7], [1, 2, 4], [5, 6, 77], [2, 3, 4]])
    fisher_vector = FisherVector(nclasses=3)
    print(fisher_vector.predict(my_data))
