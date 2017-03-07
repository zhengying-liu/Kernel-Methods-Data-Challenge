from scipy import linalg
import matplotlib.pyplot as plt
import numpy

class KernelPCA:
    def __init__(self, kernel):
        self.kernel = kernel

    def _center(self, X):
        n = X.shape[0]
        U = numpy.ones((n, n)) / n
        K = self.kernel.build_K(X)
        return numpy.dot(numpy.dot(numpy.eye(n) - U, K), numpy.eye(n) - U)

    """
    plot: if True, then plot the percentage of variance that is explained by the
        top components
    """
    def fit(self, X, plot=False):
        print("Fit PCA")
        n = X.shape[0]
        self.K = self._center(X)
        print("Calculating eigenvalue decomposition")
        eig_values, eig_vectors = linalg.eigh(self.K)

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

        if plot:
            n, m = self.alpha.shape
            variance = numpy.trace(self.K) / n
            projection = numpy.dot(self.K, self.alpha)

            percentage_explained = numpy.zeros(m)
            for i in range(m):
                percentage_explained[i] = numpy.linalg.norm(projection[:, :i]) ** 2 / n / variance * 100.0

            plt.figure()
            plt.plot(percentage_explained)

    def predict(self, components):
        assert components > 0 and components < self.alpha.shape[1]
        print("Predict PCA")
        n = self.K.shape[0]
        return numpy.dot(self.K, self.alpha[:, :components])

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
    kpca.fit(X, plot=True)
    Xproj = kpca.predict(2)
    axarr[1, 0].scatter(Xproj[y==0, 0], numpy.zeros(500), color='red')
    axarr[1, 0].scatter(Xproj[y==1, 0], numpy.zeros(500), color='blue')

    axarr[1, 1].scatter(Xproj[y==0, 0], Xproj[y==0, 1], color='red')
    axarr[1, 1].scatter(Xproj[y==1, 0], Xproj[y==1, 1], color='blue')

    plt.show()
