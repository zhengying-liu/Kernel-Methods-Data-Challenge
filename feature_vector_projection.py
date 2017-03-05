import numpy
from scipy import linalg

from kernels import build_K

class FeatureVectorProjection:
    """
    kernel: kernel that we want to estimate
    basis: basis used to calculate the projection
    G: projection matrix
    dim: number of dimensions of the projection
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.basis = None
        self.G = None
        self.ndim = None

    def fit(self, X):
        assert X.ndim == 2
        Kzz = build_K(X, self.kernel)
        Kzz = linalg.inv(Kzz)
        G = linalg.cholesky(Kzz)
        assert numpy.isreal(G).all()
        self.basis = X
        self.ndim = X.shape[0]
        self.G = G.real

    def predict(self, X):
        assert X.ndim == 2
        n, dX = X.shape
        m, dB = self.basis.shape
        assert dX == dB
        K = self.kernel.build_K(X, self.basis)
        return numpy.dot(K, self.G)

if __name__ == '__main__':
    from kernels import GaussianKernel

    sigma = 1
    kernel = GaussianKernel(sigma)
    projector = FeatureVectorProjection(kernel)
    basis = numpy.linspace(-10, 10, 20)
    basis = basis[:, numpy.newaxis]
    projector.fit(basis)

    x = numpy.linspace(-10, 10, 500)
    x = x[:, numpy.newaxis]

    features = projector.predict(x)
    center = numpy.array([0])
    center = center[:, numpy.newaxis]
    features_center = projector.predict(center)

    ypred = numpy.dot(features, features_center.T)
    y = numpy.exp(-x**2 / (2 * sigma**2))

    import matplotlib.pyplot as plt
    plt.plot(x, y, 'r', x, ypred, 'b')
    plt.show()
