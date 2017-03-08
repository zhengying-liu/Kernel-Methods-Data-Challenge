import numpy
from tqdm import tqdm

from fisher_vector import FisherVector
from gmm import Gmm
import load_features

class FisherFeatureExtractor:
    """
    local_feature_extractor_name: can be either 'hog' or 'sift'
    nclasses: number of classes used in gmm and fisher vector
    """
    def __init__(self, local_feature_extractor_name, nclasses=256):
        self.local_feature_extractor_name = local_feature_extractor_name
        self.nclasses = nclasses
        
    def train(self, X):
        assert X.ndim == 4
        print("Extracting Fisher features on training data")
        n = X.shape[0]
        ret = []
        
        local_feature_extractor = load_features.get_feature_extractor(self.local_feature_extractor_name)
        local_features = local_feature_extractor.predict(X, unflatten=True)
        
        gmm = Gmm(nclasses=self.nclasses)
        gmm.fit(local_features.reshape(-1, local_features.shape[-1]), niter=20)
        fisher_vector = FisherVector(self.nclasses, len(local_features[0, 0]), gmm.pi, gmm.mu, gmm.sigma)

        for i in tqdm(range(n)):
            ret.append(fisher_vector.predict(local_features[i,:,:]))

        return numpy.array(ret), gmm
    
    def predict(self, X, gmm):
        assert X.ndim == 4
        print("Extracting Fisher features on testing data")
        n = X.shape[0]
        ret = []
        
        local_feature_extractor = load_features.get_feature_extractor(self.local_feature_extractor_name)
        local_features = local_feature_extractor.predict(X, unflatten=True)
        
        fisher_vector = FisherVector(self.nclasses, len(local_features[0, 0]), gmm.pi, gmm.mu, gmm.sigma)

        for i in tqdm(range(n)):
            ret.append(fisher_vector.predict(local_features[i,:,:]))

        return numpy.array(ret)
