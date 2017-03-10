"""
K-means algorithm, which serves as the initialization of GMM.
"""

from tqdm import tqdm
import numpy

class Kmeans:
    """
    nclusters: number of clusters (assumed between 0 and nclusters - 1)
    dim: dimension of the space in which the data lives
    z: assignment
    mu: centers of different clusters
    """
    def __init__(self, nclusters):
        self.nclusters = nclusters
        self.dim = None
        self.z = None
        self.mu = None

    def _assignCluster(self, data):
        n_data = len(data)
        for i in tqdm(range(n_data)):
            dist_min = numpy.linalg.norm(data[i, :] - self.mu[0, :])
            for k in range(self.nclusters):
                dist = numpy.linalg.norm(data[i, :] - self.mu[k, :])
                if dist < dist_min:
                    dist_min = dist
                    self.z[i] = int(k)

    def _updateCenter(self, data):
        n_data = len(data)
        count = numpy.zeros(self.nclusters)
        self.mu = numpy.zeros((self.nclusters, self.dim))
        for i in range(n_data):
            count[self.z[i]] += 1
            self.mu[self.z[i], :] += data[i, :]
        for k in range(self.nclusters):
            if count[k] > 0:
                self.mu[k, :] /= count[k]

    def fit(self, data, niter=10, warm_start=False):
        print("Fitting K-Means on local image descriptors")
        if not(warm_start):
            self.dim = len(data[0])
            self.z = numpy.zeros(data.shape[0], dtype=int)
            index = numpy.random.random_integers(0, data.shape[0] - 1, (self.nclusters))
            self.mu = numpy.zeros((self.nclusters, self.dim))
            for i, idx in enumerate(index):
                self.mu[i] = data[idx, :]

        for _ in tqdm(range(niter)):
            self._assignCluster(data)
            self._updateCenter(data)

    def predict(self, X):
        y = numpy.zeros(X.shape[0], dtype=int)
        count = numpy.zeros(self.nclusters)
        for i, x in enumerate(X):
            dist_min = numpy.linalg.norm(x - self.mu[0, :])
            for k in range(1, self.nclusters):
                dist = numpy.linalg.norm(x - self.mu[k, :])
                if dist < dist_min:
                    y[i] = k
                    dist_min = dist
            count[ y[i] ] += 1
        print(count)
        return y

if __name__ == '__main__':
    my_data = numpy.array([[1, 2, 3], [4, 5, 6], [1, 2, 4], [-1, 0, -1]])
    model = Kmeans(nclusters=2)
    model.fit(data=my_data, niter=3)
    print(model.z)
    print(model.mu)
    print(model.predict(my_data))
