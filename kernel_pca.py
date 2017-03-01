import numpy

from kernels import build_K

class KernelPCA:
    def __init__(self, K_function):
        self.K_function = K_function

    def _center(self, X):
        n = X.shape[0]
        U = numpy.ones((n, n)) / n
        K = build_K(X, self.K_function)
        return (numpy.eye(n) - U) * K * (numpy.eye(n) - U)

    def fit(self, X):
        print("Fit PCA")
        n = X.shape[0]
        self.K = self._center(X)
        print("Calculating eigenvalue decomposition")
        eig_values, eig_vectors = numpy.linalg.eigh(self.K)

        # eigenvalues in dereasing order
        index = range(n)[::-1]
        eig_values = eig_values[index]
        eig_vectors = eig_vectors[:, index]

        self.alpha = eig_vectors
        for i, v in enumerate(eig_values):
            self.alpha[:, i] /= numpy.sqrt(v)

    def predict(self, components):
        print("Predict PCA")
        n = self.K.shape[0]
        return numpy.dot(self.K, self.alpha[:, :components])
