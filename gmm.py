"""
Guassian Mixture Model, which serves as an input argument of the Fisher Vector.
Special case: The covariance matrices are assumed to be diagonal matrices.
"""

import numpy as np
from kmeans import Kmeans

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

def logsumexp(v):
    """
    Log sum exponential computation
    
    Parameters
    ----------
    v: `numpy.array`, vector of negative integers
    
    Returns
    -------
    `float`, log(sum(exp(v)))
    """
    max_v = max(v)
    new_v = np.empty((len(v)))
    for i in range(len(v)):
        new_v[i] = v[i] - max_v
    return max_v + np.log(sum(np.exp(new_v)))

def li(x, pi, mu, sigma):
    """
    Parameters
    ----------
    x: `numpy.array`
    pi: `float`
    mu: `numpy.matrix`, a column matrix, same length as x
    sigma: `numpy.matrix`, positive definite matrix
    
    Returns
    -------
    `float`, log(pi Normal(x; mu, sigma))
    """
    d = len(x)
    x_temp = np.matrix(x.reshape(d, 1)) - mu
    sigma_inv = np.linalg.inv(sigma)
    return np.log(pi) - 0.5 * d * np.log(2 * np.pi) - 0.5 * sum(np.log(np.linalg.eigvals(sigma))) - 0.5 * (x_temp.transpose() * sigma_inv * x_temp)[0, 0]

def gamma(x, pi, mu, sigma):
    """
    Parameters
    ----------
    x: `numpy.array`, length = d
    pi: `numpy.array`, length = K
    mu: `numpy.matrix` 2d, mu[i, :] = ith column, shape = (K, d)
    sigma: `np.array of numpy.matrix` 3d, shape = (K, d, d)
    
    Returns
    -------
    `float`, sum_{j = 1}^{K} pi_j Normal(x; mu_j, sigma_j)
    `numpy.array`, the ith term  = pi_i Normal(x; mu_i, sigma_i) / sum_{j = 1}^{K} pi_j Normal(x; mu_j, sigma_j)
    """
    K = len(sigma)
    l = np.empty(K)
    for i in range(K):
        l[i] = li(x, pi[i], mu[i, :], np.matrix(sigma[i]))
    logsum = logsumexp(l)
    
    result = np.empty(K)
    for i in range(K):
        result[i] = np.exp(l[i] - logsum)
    return np.exp(logsum), result

if __name__ == '__main__':
    my_data = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 4], [5, 6, 77]])
    model = Gmm(nclasses=2)
    model.apply(data=my_data, niter=10)
    print(model.dim)
    print(model.pi)
    print(model.mu)
    print(model.sigma)
