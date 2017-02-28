import numpy

class LinearKernel:
    def __init__(self):
        pass

    def calc(self, x, y):
        return numpy.dot(x, y)

class GaussianKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def calc(self, x, y):
        return numpy.exp(-numpy.linalg.norm(x - y) ** 2) / (2 * self.sigma ** 2)
