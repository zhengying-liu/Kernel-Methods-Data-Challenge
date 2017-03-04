"""
Some utility functions to compute gamma, which is used in Gaussian Mixture Model and Fisher Vector.
"""

import numpy

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
    new_v = numpy.empty((len(v)))
    for i in range(len(v)):
        new_v[i] = v[i] - max_v
    return max_v + numpy.log(sum(numpy.exp(new_v)))

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
    x_temp = numpy.matrix(x.reshape(d, 1)) - mu
    sigma_inv = numpy.linalg.inv(sigma)
    return numpy.log(pi) - 0.5 * d * numpy.log(2 * numpy.pi) - 0.5 * sum(numpy.log(numpy.linalg.eigvals(sigma))) - 0.5 * (x_temp.transpose() * sigma_inv * x_temp)[0, 0]

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
    l = numpy.empty(K)
    for i in range(K):
        l[i] = li(x, pi[i], mu[i, :], numpy.matrix(sigma[i]))
    logsum = logsumexp(l)
    
    result = numpy.empty(K)
    for i in range(K):
        result[i] = numpy.exp(l[i] - logsum)
    return numpy.exp(logsum), result
