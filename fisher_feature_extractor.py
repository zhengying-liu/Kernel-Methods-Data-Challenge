import numpy
from tqdm import tqdm

from fisher_vector import FisherVector
from hog_feature_extractor import HOGFeatureExtractor
from sift_feature_extractor import SIFTFeatureExtractor

class FisherFeatureExtractor:
    """
    nbins: number of bins that will be used
    unsigned: if True the sign of the angle is not considered
    """
    def __init__(self, local_feature_extractor='hog', nclasses=10):
        self.local_feature_extractor = local_feature_extractor
        self.nclasses = nclasses
        
    def predict(self, X):
        assert X.ndim == 4
        print("Extracting Fisher features")
        n = X.shape[0]
        ret = []
        
        local_features = None
        if self.local_feature_extractor == 'hog':
            hog = HOGFeatureExtractor()
            local_features = hog.predict(X, unflatten=True)
        elif self.local_feature_extractor == 'sift':
            sift = SIFTFeatureExtractor()
            local_features = sift.predict(X, unflatten=True)
        else:
            raise Exception("Unknown local feature extractor")
        
        fisher_vector = FisherVector(self.nclasses)

        for i in tqdm(range(n)):
            ret.append(fisher_vector.predict(local_features[i,:,:]))

        return numpy.array(ret)
