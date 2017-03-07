"""
Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform (SIFT) algorithm by D. Lowe.
Lowe, D. G., “Distinctive Image Features from Scale-Invariant Keypoints”, International Journal of Computer Vision, 60, 2, pp. 91-110, 2004.
https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

The implementation is adapted from OpenCV.
http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
https://github.com/opencv/opencv/blob/2.4/modules/nonfree/src/sift.cpp
"""

import numpy

from image_utils import gaussian_blur, inv_transform_image_linear

# assumed gaussian blur for input image
SIFT_INIT_SIGMA = 0.5

class SIFT:
    """
    nfeatures: The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
    noctave_layers: The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
    contrast_threshold: The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
    edge_threshold: The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
    sigma: The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    """
    def __init__(self, nfeatures=0, noctave_layers=3, contrast_threshold=0.04, edge_threshold=10, sigma=1.6):
        self.nfeatures = nfeatures
        self.noctaves = -1
        self.noctave_layers = noctave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.base_image = None
        self.gaussian_pyramid = None
        self.dog_pyramid = None
        self.keypoints = None
        self.descriptors = None
    
    def _create_initial_image(self, I, sigma):
        grayI = numpy.empty((I.shape[0], I.shape[1]))
        for x in range(I.shape[0]):
            for y in range(I.shape[1]):
                grayI[x, y] = I[x, y, 0] + I[x, y, 1] + I[x, y, 2]
        
        # Double the size
        result = inv_transform_image_linear(grayI, I.shape[0] * 2, I.shape[1] * 2, 0.5, 0, 0, 0)
        
        sig_diff = numpy.sqrt(max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01))
        self.base_image = gaussian_blur(result, sig_diff)
    
    def _build_gaussian_pyramid(self):
        sig = numpy.empty(self.noctave_layers + 3)
        self.gaussian_pyramid = numpy.empty(self.noctave * (self.noctave_layers + 3))
        
        sig[0] = self.sigma
        k = numpy.power(2., 1. / self.noctave_layers)
        for i in range(1, self.noctave_layers + 3):
            sig_prev = numpy.power(k, i - 1) * self.sigma
            sig_total = sig_prev * k
            sig[i] = numpy.sqrt(sig_total ** 2 - sig_prev ** 2)

        for o in range(self.noctaves):
            for i in range(self.noctave_layers + 3):
                dst = o * (self.noctave_layers + 3) + i
                if o == 0 and i == 0:
                    self.gaussian_pyramid[dst] = self.base_image
                # base of new octave is halved image from end of previous octave
                elif i == 0:
                    src = (o - 1) * (self.noctave_layers + 3) + self.noctave_layers;
                    self.gaussian_pyramid[dst] = inv_transform_image_linear(
                        self.gaussian_pyramid[src],
                        self.gaussian_pyramid[src].shape[0] / 2,
                        self.gaussian_pyramid[src].shape[1] / 2,
                        2, 0, 0, 0)
                else:
                    src = o * (self.noctave_layers + 3) + i - 1
                    self.gaussian_pyramid[dst] = gaussian_blur(self.gaussian_pyramid[src], sig[i])
    
    def _build_DoG_pyramid(self):
        self.dog_pyramid = numpy.empty(self.noctaves * (self.noctave_layers + 2))

        for o in range(self.noctaves):
            for i in range(self.noctave_layers + 2):
                src1 = o * (self.noctave_layers + 3) + i
                src2 = src1 + 1
                dst = o * (self.noctave_layers + 2) + i
                self.dog_pyramid[dst] = self.gaussian_pyramid[src2] - self.gaussian_pyramid[src1]
    
    def _calc_orientation_hist(self, I, x, y, radius, hist, n):
        pass
    
    def _adjust_local_extrema(self, octave, layer, x, y):
        pass
    
    def _find_scale_space_extrema(self):
        pass
    
    def _calc_SIFT_descriptor(self):
        pass
    
    def _calc_descriptors(self):
        pass
    
    def calc_features_for_image(self, I):
        self.noctaves = int(numpy.round(numpy.log2(min(I.shape[0], I.shape[1])))) - 2
    