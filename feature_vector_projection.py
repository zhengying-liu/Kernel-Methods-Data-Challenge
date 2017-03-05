import numpy

from kernels import build_K

class FeatureVectorProjection:
    """
    kernel: kernel that we want to estimate
    basis: basis used to calculate the projection
    G: projection matrix
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.basis = None
        self.G = None

    def fit(self, X):
        Kzz = build_K(X, self.kernel)
        Kzz = numpy.linalg.inv(Kzz)
        G = numpy.linalg.cholesky(Kzz)
        assert numpy.isreal(G).all()
        self.basis = X
        self.G = G.real

    def predict(self, X):
        n, dX = X.shape
        m, dB = self.basis.shape
        assert dX == dB
        K = numpy.zeros((n, m))
        for i in range(n):
            x = X[i, :]
            for j in range(m):
                K[i, j] = self.kernel.calc(x, self.basis[j, :])
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
