import numpy
from tqdm import tqdm

from sift import SIFT

class SIFTFeatureExtractor:
    """
    nfeatures: The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
    noctave_layers: The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
    contrast_threshold: The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
    edge_threshold: The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
    sigma: The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    """
    def __init__(self, nfeatures=10, noctave_layers=3, contrast_threshold=0.001, edge_threshold=10, sigma=1.6):
        self.nfeatures = nfeatures
        self.noctave_layers = noctave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma

    def predict(self, X, unflatten=False):
        assert X.ndim == 4
        print("Extracting SIFT features")
        n = X.shape[0]
        ret = []

        for i in tqdm(range(n)):
            sift = SIFT(self.nfeatures, self.noctave_layers, self.contrast_threshold, self.edge_threshold, self.sigma)
            ret.append(sift.calc_features_for_image(X[i,:,:,:], unflatten))
        
        if not unflatten:
            return numpy.array(ret)
        return ret
    