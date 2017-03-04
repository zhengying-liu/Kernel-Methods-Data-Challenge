"""
Guassian Mixture Model, which serves as an input argument of the Fisher Vector.
Special case: The covariance matrices are assumed to be diagonal matrices.
"""

import numpy as np

from kmeans import Kmeans
from gamma import gamma

class Gmm:
    """
    nclasses: number of classes (assumed between 0 and nclasses - 1)
    dim: dimension of the space in which the data lives
    pi: proportion of different classes
    mu: centers of different classes
    sigma: diagonal values of the covariance matrix, represented by a 2d numpy array
    """

    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.dim = None
        self.pi = np.ones(shape=(self.nclasses)) * (1. / self.nclasses)
        self.mu = None
        self.sigma = None
    
    def _kmeans_init(self, data):
        self.dim = len(data[0])
        kmeans = Kmeans(nclusters=self.nclasses)
        kmeans.apply(data=data, niter=10)
        self.mu = kmeans.mu
        self.sigma = np.ones(shape=(self.nclasses, self.dim))

    def _EM_iterate_once(self, data):
        N = len(data)
        K = self.nclasses
        gamma_K_N = np.empty((K, N))
        
        sigma_matrix = np.zeros((K, self.dim, self.dim))
        for k in range(K):
            sigma_matrix[k] = np.diag(self.sigma[k])
            
        for n in range(N):
            gamma_K_N[:,n] = gamma(data[n], self.pi, self.mu, sigma_matrix)[1]
        
        gamma_sum_K = np.empty(K)
        for i in range(K):
            gamma_sum_K[i] = sum(gamma_K_N[i])
        
        pi_new = np.empty(K)
        for i in range(K):
            pi_new[i] = 1. / N * gamma_sum_K[i]
        
        mu_new = np.dot(np.matrix(gamma_K_N), np.matrix(data))
        for i in range(K):
            mu_new[i, :] /= gamma_sum_K[i]
        
        sigma_new = np.zeros((K, self.dim))
        for i in range(K):
            for n in range(N):
                x_temp = np.matrix(data[n].reshape(self.dim, 1)) - mu_new[i, :]
                sigma_new[i] += gamma_K_N[i][n] * np.diag(x_temp * x_temp.transpose())
            sigma_new[i] /= gamma_sum_K[i]
            
        self.pi = pi_new
        self.mu = mu_new
        self.sigma = sigma_new
        
    def apply(self, data, niter=10):
        self._kmeans_init(data)
        for _ in range(niter):
            self._EM_iterate_once(data)

if __name__ == '__main__':
    my_data = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 4], [5, 6, 77]])
    model = Gmm(nclasses=2)
    model.apply(data=my_data, niter=10)
    print(model.dim)
    print(model.pi)
    print(model.mu)
    print(model.sigma)
