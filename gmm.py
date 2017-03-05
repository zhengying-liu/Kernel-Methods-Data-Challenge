"""
Guassian Mixture Model, which serves as an input argument of the Fisher Vector.
Special case: The covariance matrices are assumed to be diagonal matrices.
"""

import numpy

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
        self.pi = numpy.ones(shape=(self.nclasses)) * (1. / self.nclasses)
        self.mu = None
        self.sigma = None
    
    def _kmeans_init(self, data):
        self.dim = len(data[0])
        kmeans = Kmeans(nclusters=self.nclasses)
        kmeans.fit(data=data, niter=10)
        self.mu = kmeans.mu
        self.sigma = numpy.ones(shape=(self.nclasses, self.dim))

    def _EM_iterate_once(self, data):
        N = len(data)
        K = self.nclasses
        gamma_K_N = numpy.empty((K, N))
        
        sigma_matrix = numpy.zeros((K, self.dim, self.dim))
        for k in range(K):
            sigma_matrix[k] = numpy.diag(self.sigma[k])
            
        for n in range(N):
            gamma_K_N[:,n] = gamma(data[n], self.pi, self.mu, sigma_matrix)[1]
        
        gamma_sum_K = numpy.sum(gamma_K_N, axis=1)
        
        pi_new = 1. / N * gamma_sum_K
        
        mu_new = numpy.dot(numpy.matrix(gamma_K_N), numpy.matrix(data))
        for i in range(K):
            mu_new[i, :] /= gamma_sum_K[i]
        
        sigma_new = numpy.zeros((K, self.dim))
        for i in range(K):
            for n in range(N):
                x_temp = numpy.matrix(data[n].reshape(self.dim, 1)) - mu_new[i, :]
                sigma_new[i] += gamma_K_N[i][n] * numpy.diag(x_temp * x_temp.transpose())
            sigma_new[i] /= gamma_sum_K[i]
            
        self.pi = pi_new
        self.mu = mu_new
        self.sigma = sigma_new
        
    def fit(self, data, niter=10):
        self._kmeans_init(data)
        for _ in range(niter):
            self._EM_iterate_once(data)

if __name__ == '__main__':
    my_data = numpy.array([[1, 2, 3], [4, 5, 6], [1, 2, 4], [5, 6, 77]])
    model = Gmm(nclasses=2)
    model.fit(data=my_data, niter=10)
    print(model.dim)
    print(model.pi)
    print(model.mu)
    print(model.sigma)
