import numpy

from hog_feature_extractor import HOGFeatureExtractor
from kmeans import Kmeans

class BagOfWords:
    def __init__(self, local_feature_extractor_name='hog', nclusters=256):
        self.nclusters = nclusters
        if local_feature_extractor_name == 'hog':
            self.feature_extractor = HOGFeatureExtractor()
        else:
            raise Exception("Unknown feature extractor")
        self.kmeans = None

    def extract(self, X):
        assert X.ndim == 4
        return self.feature_extractor.predict(X, unflatten=True)

    def fit(self, X):
        assert X.ndim == 3
        X_features = X.reshape(X.shape[0] * X.shape[1], -1)
        self.kmeans = Kmeans(self.nclusters)
        self.kmeans.fit(X_features)

    def predict(self, X):
        assert X.ndim == 3
        X_features = X.reshape(X.shape[0] * X.shape[1], -1)
        X_clustered = self.kmeans.predict(X_features)
        X_clustered = X_clustered.reshape(X.shape[0], X.shape[1])
        ret = numpy.zeros((X.shape[0], self.nclusters))

        for i, x in enumerate(X_clustered):
            for word in x:
                ret[i, word] += 1

        return ret
