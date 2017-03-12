import matplotlib.pyplot as plt
import numpy
from scipy import linalg

class KernelPCA:
    def __init__(self, kernel):
        self.kernel = kernel
        self.X = None
        self.alpha = None
        self.center_vector = None

    def _center(self, X):
        n = X.shape[0]
        U = numpy.ones((n, n)) / n
        K = self.kernel.build_K(X)
        self.center_vector = numpy.mean(K, axis=1) - numpy.mean(K)
        return numpy.dot(numpy.dot(numpy.eye(n) - U, K), numpy.eye(n) - U)

    """
    cut_percentage: percentage of variance that needs to be explained by the
        components, if None, we keep all components
    plot: if True, then plot the percentage of variance that is explained by the
        top components
    """
    def fit(self, X, cut_percentage=None, plot=False):
        print("Fit KPCA")
        n = X.shape[0]
        self.X = X
        K = self._center(X)
        print("Calculating eigenvalue decomposition")
        eig_values, eig_vectors = linalg.eigh(K)

        # eigenvalues in dereasing order
        index = range(n)[::-1]
        eig_values = eig_values[index]
        eig_vectors = eig_vectors[:, index]

        # filter non-positive eigenvalues
        index = eig_values > 0
        eig_values = eig_values[index]
        eig_vectors = eig_vectors[:, index]

        self.alpha = eig_vectors
        for i, v in enumerate(eig_values):
            self.alpha[:, i] /= numpy.sqrt(v)

        if cut_percentage is not None:
            assert cut_percentage > 0 and cut_percentage <= 100

            n, m = self.alpha.shape
            variance = numpy.trace(K) / n
            projection = numpy.dot(K, self.alpha)
            components = self.alpha.shape[1]

            for i in range(m):
                percentage_explained = numpy.linalg.norm(projection[:, :i + 1]) ** 2 / n / variance * 100.0
                if percentage_explained >= cut_percentage:
                    print i + 1, percentage_explained
                    components = i + 1
                    break

            self.alpha = self.alpha[:, :components]

        if plot:
            n, m = self.alpha.shape
            variance = numpy.trace(K) / n
            projection = numpy.dot(K, self.alpha)

            percentage_explained = numpy.zeros(m)
            for i in range(m):
                percentage_explained[i] = numpy.linalg.norm(projection[:, :i + 1]) ** 2 / n / variance * 100.0

            plt.figure()
            plt.plot(percentage_explained)

    def predict(self, X, components=None):
        assert components is None or (components > 0 and components <= self.alpha.shape[1])
        if components is None:
            components = self.alpha.shape[1]

        K = self.kernel.build_K(X, self.X)
        K = K - numpy.mean(K, axis=1)[:, numpy.newaxis] - self.center_vector
        return numpy.dot(K, self.alpha[:, :components])

if __name__ == '__main__':
    from sklearn.datasets import make_circles
    from kernels import LinearKernel, GaussianKernel

    f, axarr = plt.subplots(2, 2, sharex=True)

    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    axarr[0, 0].scatter(X[y==0, 0], X[y==0, 1], color='red')
    axarr[0, 0].scatter(X[y==1, 0], X[y==1, 1], color='blue')

    kpca = KernelPCA(LinearKernel())
    kpca.fit(X)
    Xproj = kpca.predict(1)
    axarr[0, 1].scatter(Xproj[y==0, 0], numpy.zeros(500), color='red')
    axarr[0, 1].scatter(Xproj[y==1, 0], numpy.zeros(500), color='blue')

    # decrease sigma to improve separation
    kpca = KernelPCA(GaussianKernel(0.686))
    kpca.fit(X, cut_percentage=95, plot=True)
    print kpca.alpha.shape[1]
    Xproj = kpca.predict(2)
    axarr[1, 0].scatter(Xproj[y==0, 0], numpy.zeros(500), color='red')
    axarr[1, 0].scatter(Xproj[y==1, 0], numpy.zeros(500), color='blue')

    axarr[1, 1].scatter(Xproj[y==0, 0], Xproj[y==0, 1], color='red')
    axarr[1, 1].scatter(Xproj[y==1, 0], Xproj[y==1, 1], color='blue')

    plt.show()
