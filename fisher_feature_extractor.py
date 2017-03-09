import numpy
from tqdm import tqdm

from fisher_vector import FisherVector
from gmm import Gmm
import load_features
from pca import pca 

n_components = 16
# n_components = 3

class FisherFeatureExtractor:
    """
    local_feature_extractor_name: can be either 'hog' or 'sift'
    nclasses: number of classes used in gmm and fisher vector
    """
    def __init__(self, local_feature_extractor_name, nclasses=256, kmeans_niter=10, gmm_niter=10):
        self.local_feature_extractor_name = local_feature_extractor_name
        self.nclasses = nclasses
        self.kmeans_niter = kmeans_niter
        self.gmm_niter = gmm_niter
        
    def train(self, X):
        assert X.ndim == 4
        print("Extracting Fisher features on training data")
        n = X.shape[0]
        ret = []
        
        local_feature_extractor = load_features.get_feature_extractor(self.local_feature_extractor_name)
        local_features = local_feature_extractor.predict(X, unflatten=True)
        
        _, V_truncate = pca(local_features.reshape(-1, local_features.shape[-1]), components=n_components)
        local_features_pca = []
        for i in range(n):
            local_features_pca.append(numpy.array(numpy.matrix(local_features[i,:,:]) * V_truncate))
        local_features_pca = numpy.array(local_features_pca)
        
        gmm = Gmm(nclasses=self.nclasses)
        gmm.fit(local_features_pca.reshape(-1, local_features_pca.shape[-1]), kmeans_niter=self.kmeans_niter, niter=self.gmm_niter)
        fisher_vector = FisherVector(self.nclasses, len(local_features_pca[0, 0]), gmm.pi, gmm.mu, gmm.sigma)

        for i in tqdm(range(n)):
            ret.append(fisher_vector.predict(local_features_pca[i,:,:]))

        return numpy.array(ret), V_truncate, gmm
    
    def predict(self, X, V_truncate, gmm):
        assert X.ndim == 4
        print("Extracting Fisher features on testing data")
        n = X.shape[0]
        ret = []
        
        local_feature_extractor = load_features.get_feature_extractor(self.local_feature_extractor_name)
        local_features = local_feature_extractor.predict(X, unflatten=True)
        
        local_features_pca = []
        for i in range(n):
            local_features_pca.append(numpy.array(numpy.matrix(local_features[i,:,:]) * V_truncate))
        local_features_pca = numpy.array(local_features_pca)
        
        fisher_vector = FisherVector(self.nclasses, len(local_features_pca[0, 0]), gmm.pi, gmm.mu, gmm.sigma)

        for i in tqdm(range(n)):
            ret.append(fisher_vector.predict(local_features_pca[i,:,:]))

        return numpy.array(ret)
