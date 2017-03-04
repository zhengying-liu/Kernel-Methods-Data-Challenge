from tqdm import tqdm
import numpy

def build_K(X, K_function):
    print("Buliding kernel matrix")
    n = X.shape[0]
    K = numpy.zeros((n, n))

    for i in tqdm(range(n)):
        for j in range(n):
            K[i, j] = K_function.calc(X[i, :], X[j, :])

    return K

class LinearKernel:
    def calc(self, x, y):
        return numpy.dot(x, y)

class GaussianKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def calc(self, x, y):
        return numpy.exp(-numpy.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

class HistogramIntersectionKernel:
    def __init__(self, beta):
        self.beta = beta

    def calc(self, x, y):
        return numpy.sum(numpy.minimum(x ** self.beta, y ** self.beta))

class LaplacianRBFKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def calc(self, x, y):
        return numpy.exp(-numpy.sum(numpy.abs(x - y)) / self.sigma**2)

class SublinearRBFKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def calc(self, x, y):
        return numpy.exp(-numpy.sum(numpy.abs(x - y))**0.5 / self.sigma**2)
