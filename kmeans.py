"""
K-means algorithm, which serves as the initialization of GMM.
"""

import numpy as np

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
        self.z = np.zeros(n_data, dtype=int)
        for i in range(n_data):
            dist_min = np.linalg.norm(data[i, :] - np.array(self.mu[0, :])[0])
            for k in range(self.nclusters):
                dist = np.linalg.norm(data[i, :] - np.array(self.mu[k, :])[0])
                if dist < dist_min:
                    dist_min = dist
                    self.z[i] = int(k)
    
    def _updateCenter(self, data):
        n_data = len(data)
        count = np.zeros(self.nclusters)
        self.mu = np.matrix(np.zeros((self.nclusters, self.dim)))
        for i in range(n_data):
            count[self.z[i]] += 1
            for d in range(self.dim):
                self.mu[self.z[i], d] += data[i, d]
        for k in range(self.nclusters):
            for d in range(self.dim):
                if count[k] != 0:
                    self.mu[k, d] /= count[k]
    
    def fit(self, data, niter=1, warm_start=False):
        if not(warm_start):
            self.dim = len(data[0])
            self.mu = np.matrix(np.zeros((self.nclusters, self.dim)))
        for _ in range(niter):
            self._assignCluster(data)
            self._updateCenter(data)
    
if __name__ == '__main__':
    my_data = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 4], [-1, 0, -1]])
    model = Kmeans(nclusters=2)
    model.fit(data=my_data, niter=3)
    print(model.z)
    print(model.mu)