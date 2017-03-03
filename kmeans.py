"""
K-means
"""

import numpy as np

class Kmeans:
    """
    nclasses: number of classes (assumed between 0 and nclasses - 1)
    dim: dimension of the space in which the data lives
    z: assignment
    """

    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.dim = None
        self.z = None
        self.mu = None
        
    def _assignCluster(self, data, n_data):
        self.z = np.zeros(n_data, dtype=int)
        for i in range(n_data):
            dist_min = np.linalg.norm(data[i, :] - np.array(self.mu[0, :])[0])
            for k in range(self.nclasses):
                dist = np.linalg.norm(data[i, :] - np.array(self.mu[k, :])[0])
                if dist < dist_min:
                    dist_min = dist
                    self.z[i] = int(k)
    
    def _updateCenter(self, data, n_data):
        count = np.zeros(self.nclasses)
        self.mu = np.matrix(np.zeros((self.nclasses, self.dim)))
        for i in range(n_data):
            count[self.z[i]] += 1
            for d in range(self.dim):
                self.mu[self.z[i], d] += data[i, d]
        for k in range(self.nclasses):
            for d in range(self.dim):
                if count[k] != 0:
                    self.mu[k, d] /= count[k]
    
    def apply(self, steps, dim, data, n_data, warm_start=False):
        if not(warm_start):
            self.dim = dim
            self.mu = np.matrix(np.zeros((self.nclasses, self.dim)))
        for _ in range(steps):
            self._assignCluster(data, n_data)
            self._updateCenter(data, n_data)
    
if __name__ == '__main__':
    my_data = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 4], [-1, 0, -1]])
    model = Kmeans(2)
    model.apply(steps=3, dim=3, data=my_data, n_data=4)
    print(model.z)
    print(model.mu)
    
