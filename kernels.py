from tqdm import tqdm
import numpy

class Kernel:
    def __init__():
        self.name = None

    def calc(self, x, y):
        raise NotImplementedError("calc function has not been implemented")

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        K = numpy.zeros((n, m))

        for i in tqdm(range(n)):
            for j in range(m):
                K[i, j] = self.calc(X[i, :], Y[j, :])
        return K

class LinearKernel(Kernel):
    def __init__(self):
        self.name = 'linear'

    def calc(self, x, y):
        return numpy.dot(x, y)

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        return numpy.dot(X, Y.T)

class GaussianKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'gaussian_%.5f' % sigma

    def calc(self, x, y):
        return numpy.exp(-numpy.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        K = numpy.zeros((n, m))

        for i in tqdm(range(m)):
            K[:, i] = numpy.linalg.norm(X - Y[i, :], axis=1) ** 2
        K /= 2 * self.sigma ** 2
        return numpy.exp(-K)

class GaussianKernelForAngle(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'gaussian_angle_%.5f' % sigma

    def calc(self, x, y):
        aux = (numpy.sin(x) - numpy.sin(y)) ** 2 + (numpy.cos(x) - numpy.cos(y)) ** 2
        return numpy.exp(-aux / (2 * self.sigma ** 2))

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        X2 = numpy.concatenate((numpy.sin(X),numpy.cos(X)), axis=1)
        Y2 = numpy.concatenate((numpy.sin(Y),numpy.cos(Y)), axis=1)
        K = numpy.zeros((n, m))

        for i in tqdm(range(m)):
            K[:, i] = numpy.linalg.norm(X2 - Y2[i, :], axis=1) ** 2
        K /= 2 * self.sigma ** 2
        return numpy.exp(-K)

class HistogramIntersectionKernel(Kernel):
    def __init__(self, beta):
        self.beta = beta
        self.name = 'histogram_intersection'

    def calc(self, x, y):
        return numpy.sum(numpy.minimum(x ** self.beta, y ** self.beta))

class LaplacianRBFKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'laplacian_%.5f' % sigma

    def calc(self, x, y):
        return numpy.exp(-numpy.sum(numpy.abs(x - y)) / self.sigma**2)

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        K = numpy.zeros((n, m))

        for i in tqdm(range(m)):
            K[:, i] = numpy.sum(numpy.abs(X - Y[i, :]), axis=1)
        K /= self.sigma ** 2
        return numpy.exp(-K)

class SublinearRBFKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = 'sublinear_%.5f' % sigma

    def calc(self, x, y):
        return numpy.exp(-numpy.sum(numpy.abs(x - y))**0.5 / self.sigma**2)

class HellingerKernel(Kernel):
    def __init__(self):
        self.name = 'hellinger_%.5f' % sigma

    def calc(self, x, y):
        return numpy.sum(numpy.sqrt(x * y))
