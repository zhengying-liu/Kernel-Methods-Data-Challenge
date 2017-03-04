"""
Some utility functions to compute gamma, which is used in Gaussian Mixture Model and Fisher Vector.
"""

import numpy as np

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
