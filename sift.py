"""
Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform (SIFT) algorithm by D. Lowe.
Lowe, D. G., Distinctive Image Features from Scale-Invariant Keypoints, International Journal of Computer Vision, 60, 2, pp. 91-110, 2004.
https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

The implementation is adapted from OpenCV.
http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
https://github.com/opencv/opencv/blob/2.4/modules/nonfree/src/sift.cpp
"""

import matplotlib.pyplot as plt
import numpy

from keypoint import Keypoint
from image_utils import gaussian_blur, inv_transform_image_linear
from matplotlib.patches import Circle

plot = False

# assumed gaussian blur for input image
SIFT_INIT_SIGMA = 0.5

# default number of bins per histogram in descriptor array
SIFT_ORI_HIST_BINS = 8

# width of border in which to ignore keypoints
SIFT_IMG_BORDER = 4 # 5

# maximum steps of keypoint interpolation before failure
SIFT_MAX_INTERP_STEPS = 5

# default number of bins in histogram for orientation assignment
SIFT_ORI_HIST_BINS = 36

# determines gaussian sigma for orientation assignment
SIFT_ORI_SIG_FCTR = 1.5

# determines the radius of the region used in orientation assignment
SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR

# orientation magnitude relative to max that results in new feature
SIFT_ORI_PEAK_RATIO = 0.8

# default width of descriptor histogram array
SIFT_DESCR_WIDTH = 2 # 4

# default number of bins per histogram in descriptor array
SIFT_DESCR_HIST_BINS = 8

# determines the size of a single descriptor orientation histogram
SIFT_DESCR_SCL_FCTR = 3.

# threshold on magnitude of elements of descriptor vector
SIFT_DESCR_MAG_THR = 0.2

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
    
    def _create_initial_image(self, I):
        grayI = numpy.empty((I.shape[0], I.shape[1]))
        for x in range(I.shape[0]):
            for y in range(I.shape[1]):
                grayI[x, y] = I[x, y, 0] + I[x, y, 1] + I[x, y, 2]
        
        # Double the size
        result = inv_transform_image_linear(grayI, I.shape[0] * 2, I.shape[1] * 2, 0.5, 0, 0, 0)
        
        sig_diff = numpy.sqrt(max(self.sigma * self.sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01))
        self.base_image = gaussian_blur(result, sig_diff)
    
    def _build_gaussian_pyramid(self):
        sig = numpy.empty(self.noctave_layers + 3)
        # self.gaussian_pyramid = numpy.empty(self.noctaves * (self.noctave_layers + 3))
        self.gaussian_pyramid = []
        
        sig[0] = self.sigma
        k = numpy.power(2., 1. / self.noctave_layers)
        for i in range(1, self.noctave_layers + 3):
            sig_prev = numpy.power(k, i - 1) * self.sigma
            sig_total = sig_prev * k
            sig[i] = numpy.sqrt(sig_total ** 2 - sig_prev ** 2)

        for o in range(self.noctaves):
            for i in range(self.noctave_layers + 3):
                # dst = o * (self.noctave_layers + 3) + i
                if o == 0 and i == 0:
                    self.gaussian_pyramid.append(self.base_image)
                # base of new octave is halved image from end of previous octave
                elif i == 0:
                    src = (o - 1) * (self.noctave_layers + 3) + self.noctave_layers;
                    self.gaussian_pyramid.append(inv_transform_image_linear(
                        self.gaussian_pyramid[src],
                        self.gaussian_pyramid[src].shape[0] / 2,
                        self.gaussian_pyramid[src].shape[1] / 2,
                        2, 0, 0, 0))
                else:
                    src = o * (self.noctave_layers + 3) + i - 1
                    self.gaussian_pyramid.append(gaussian_blur(self.gaussian_pyramid[src], sig[i]))
                        
    def _build_DoG_pyramid(self):
        # self.dog_pyramid = numpy.empty(self.noctaves * (self.noctave_layers + 2))
        self.dog_pyramid = []

        for o in range(self.noctaves):
            for i in range(self.noctave_layers + 2):
                src1 = o * (self.noctave_layers + 3) + i
                src2 = src1 + 1
                # dst = o * (self.noctave_layers + 2) + i
                self.dog_pyramid.append(self.gaussian_pyramid[src2] - self.gaussian_pyramid[src1])
    
    # Computes a gradient orientation histogram at a specified pixel
    def _calc_orientation_hist(self, I, px, py, radius, weight_sigma, nbins):
        expf_scale = -1. / (2. * weight_sigma * weight_sigma)
        width = I.shape[0]
        height = I.shape[1]
        temphist = numpy.zeros(nbins)
        
        for i in range(-radius, radius + 1):
            x = px + i
            if x <= 0 or x >= width - 1:
                continue
            for j in range(-radius, radius + 1):
                y = py + j
                if y <= 0 or y >= height - 1:
                    continue
                
                dx = I[x + 1, y] - I[x - 1, y]
                dy = I[x, y + 1] - I[x, y - 1]
                
                # compute gradient values, orientations and the weights over the pixel neighborhood
                weight = numpy.exp((i * i + j * j) * expf_scale)
                angle = numpy.arctan2(dy, dx)
                mag = numpy.sqrt(dx * dx + dy * dy)
                
                binnum = int(numpy.round((nbins / (2 * numpy.pi)) * angle))
                if binnum >= nbins:
                    binnum -= nbins
                if binnum < 0:
                    binnum += nbins
                temphist[binnum] += weight * mag
         
        # smooth the histogram
        hist = numpy.zeros(nbins)
        for i in range(nbins):
            hist[i] = (temphist[(i - 2 + nbins) % nbins] + temphist[(i + 2) % nbins]) * (1. / 16.) + \
                (temphist[(i - 1 + nbins) % nbins] + temphist[(i + 1) % nbins]) * (1. / 4.) + \
                temphist[i] * (6. / 16.)
        
        return hist
    
    # Interpolates a scale-space extremum's location and scale to subpixel
    # accuracy to form an image feature. Rejects features with low contrast.
    # Based on Section 4 of Lowe's paper.
    def _adjust_local_extrema(self, octv, layer, x, y):
        di = 0
        dx = 0
        dy = 0
        finished = False
    
        for _ in range(SIFT_MAX_INTERP_STEPS):
            idx = octv * (self.noctave_layers + 2) + layer
            img = self.dog_pyramid[idx]
            prv = self.dog_pyramid[idx - 1]
            nxt = self.dog_pyramid[idx + 1]
    
            dD = numpy.array([(img[x+1, y] - img[x-1, y]) * 0.5, (img[x, y+1] - img[x, y-1]) * 0.5, (nxt[x, y] - prv[x, y]) * 0.5])
    
            v2 = img[x, y] * 2;
            dxx = img[x+1, y] + img[x-1, y] - v2
            dyy = img[x, y+1] + img[x, y-1] - v2
            dss = nxt[x, y] + prv[x, y] - v2
            dxy = (img[x+1, y+1] - img[x+1, y-1] - img[x-1, y+1] + img[x-1, y-1]) * 0.25
            dxs = (nxt[x+1, y] - nxt[x-1, y] - prv[x+1, y] + prv[x-1, y]) * 0.25
            dys = (nxt[x, y+1] - nxt[x, y-1] - prv[x, y+1] + prv[x, y-1]) * 0.25
    
            H = numpy.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
            X = numpy.linalg.solve(H, dD)
            dx = -X[0]
            dy = -X[1]
            di = -X[2]
     
            if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(di) < 0.5:
                finished = True
                break
     
            x += int(numpy.round(dx))
            y += int(numpy.round(dy))
            layer += int(numpy.round(di))
     
            if (layer < 1 or layer > self.noctave_layers or
                x < SIFT_IMG_BORDER or x >= img.shape[0] - SIFT_IMG_BORDER or
                y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER):
                return None, 0, 0, 0
     
        # ensure convergence of interpolation
        if not finished:
            return None, 0, 0, 0
     
        idx = octv * (self.noctave_layers + 2) + layer
        img = self.dog_pyramid[idx]
        prv = self.dog_pyramid[idx - 1]
        nxt = self.dog_pyramid[idx + 1]
        dD = numpy.array([(img[x+1, y] - img[x-1, y]) * 0.5, (img[x, y+1] - img[x, y-1]) * 0.5, (nxt[x, y] - prv[x, y]) * 0.5])
        t = numpy.dot(dD, numpy.array([dx, dy, di]))
 
        contr = img[x, y] + t * 0.5
        if abs(contr) * self.noctave_layers < self.contrast_threshold:
            return None, 0, 0, 0
 
        # principal curvatures are computed using the trace and det of Hessian
        v2 = img[x, y] * 2;
        dxx = img[x+1, y] + img[x-1, y] - v2
        dyy = img[x, y+1] + img[x, y-1] - v2
        dxy = (img[x+1, y+1] - img[x+1, y-1] - img[x-1, y+1] + img[x-1, y-1]) * 0.25
        tr = dxx + dyy
        det = dxx * dyy - dxy * dxy
 
        if det <= 0 or tr * tr * self.edge_threshold >= ((self.edge_threshold + 1) ** 2) * det:
            return None, 0, 0, 0
     
        kpt = Keypoint()
        kpt.x = (x + dx) * (1 << octv)
        kpt.y = (y + dy) * (1 << octv)
        kpt.octave = octv
        kpt.layer = layer + di
        kpt.sigma = self.sigma * numpy.power(2.0, (layer + di) / self.noctave_layers) * (1 << octv)
        kpt.response = abs(contr)
        return kpt, layer, x, y
    
    def _find_scale_space_extrema(self):
        threshold = 0.5 * self.contrast_threshold / self.noctave_layers
    
        self.keypoints = []
        
        ncandidate = 0
    
        for o in range(self.noctaves):
            for i in range(1, self.noctave_layers + 1):
                idx = o * (self.noctave_layers + 2) + i
                img = self.dog_pyramid[idx]
                prv = self.dog_pyramid[idx - 1]
                nxt = self.dog_pyramid[idx + 1]
                width = img.shape[0]
                height = img.shape[1]
    
                for x in range(SIFT_IMG_BORDER, width - SIFT_IMG_BORDER):
                    for y in range(SIFT_IMG_BORDER, height - SIFT_IMG_BORDER):
                        val = img[x, y]
    
                        # find local extrema with pixel accuracy
                        if abs(val) > threshold and ((
#                             val > 0 and val >= img[x, y-1] and val >= img[x, y+1] and
#                             val >= img[x-1, y-1] and val >= img[x-1, y] and val >= img[x-1, y+1] and
#                             val >= img[x+1, y-1] and val >= img[x+1, y] and val >= img[x+1, y+1] and
#                             val >= nxt[x-1, y-1] and val >= nxt[x-1, y] and val >= nxt[x-1, y+1] and
#                             val >= nxt[x, y-1] and val >= nxt[x, y] and val >= nxt[x, y+1] and
#                             val >= nxt[x+1, y-1] and val >= nxt[x+1, y] and val >= nxt[x+1, y+1] and
#                             val >= prv[x-1, y-1] and val >= prv[x-1, y] and val >= prv[x-1, y+1] and
#                             val >= prv[x, y-1] and val >= prv[x, y] and val >= prv[x, y+1] and
#                             val >= prv[x+1, y-1] and val >= prv[x+1, y] and val >= prv[x+1, y+1]) or (
#                             val < 0 and val <= img[x, y-1] and val <= img[x, y+1] and
#                             val <= img[x-1, y-1] and val <= img[x-1, y] and val <= img[x-1, y+1] and
#                             val <= img[x+1, y-1] and val <= img[x+1, y] and val <= img[x+1, y+1] and
#                             val <= nxt[x-1, y-1] and val <= nxt[x-1, y] and val <= nxt[x-1, y+1] and
#                             val <= nxt[x, y-1] and val <= nxt[x, y] and val <= nxt[x, y+1] and
#                             val <= nxt[x+1, y-1] and val <= nxt[x+1, y] and val <= nxt[x+1, y+1] and
#                             val <= prv[x-1, y-1] and val <= prv[x-1, y] and val <= prv[x-1, y+1] and
#                             val <= prv[x, y-1] and val <= prv[x, y] and val <= prv[x, y+1] and
#                             val <= prv[x+1, y-1] and val <= prv[x+1, y] and val <= prv[x+1, y+1])):
                            val > 0 and val >= img[x, y-1] and val >= img[x, y+1] and
                            val >= img[x+1, y] and val >= img[x-1, y] and
                            val >= nxt[x, y] and val >= prv[x, y]) or (
                            val < 0 and val <= img[x, y-1] and val <= img[x, y+1] and
                            val <= img[x+1, y] and val <= img[x-1, y] and
                            val <= nxt[x, y] and val <= prv[x, y])):
                            
                            ncandidate += 1
                            kpt, i2, x2, y2 = self._adjust_local_extrema(o, i, x, y)
                            if kpt is None:
                                continue
                            scl_octv = kpt.sigma / (1 << o)
                            n = SIFT_ORI_HIST_BINS
                            hist = self._calc_orientation_hist(self.gaussian_pyramid[o * (self.noctave_layers + 3) + i2],
                                                               x2,
                                                               y2,
                                                               int(numpy.round(SIFT_ORI_RADIUS * scl_octv)),
                                                               SIFT_ORI_SIG_FCTR * scl_octv, n)
                            
                            mag_threshold = numpy.max(hist) * SIFT_ORI_PEAK_RATIO
                            for j in range(n):
                                left = j - 1 if j > 0 else n - 1
                                right = j + 1 if j < n - 1 else 0
     
                                if hist[j] > hist[left] and hist[j] > hist[right] and hist[j] >= mag_threshold:
                                    binnum = j + 0.5 * (hist[left] - hist[right]) / (hist[left] - 2 * hist[j] + hist[right])
                                    binnum = binnum + n if binnum < 0 else binnum
                                    binnum = binnum - n if binnum >= n else binnum
                                    kpt.angle = (2 * numpy.pi / n) * binnum
                                    self.keypoints.append(kpt.clone())
    
    def _calc_SIFT_descriptor(self, I, xf, yf, angle, sigma):
        d = SIFT_DESCR_WIDTH
        n = SIFT_DESCR_HIST_BINS
        x = int(numpy.round(xf))
        y = int(numpy.round(yf))
        cos_t = numpy.cos(angle)
        sin_t = numpy.sin(angle)
        bins_per_rad = n / (2 * numpy.pi)
        exp_scale = -1./(d * d * 0.5)
        hist_width = SIFT_DESCR_SCL_FCTR * sigma
        radius = int(numpy.round(hist_width * 1.4142135623730951 * (d + 1) * 0.5))
        cos_t /= hist_width
        sin_t /= hist_width
        
        width = I.shape[0]
        height = I.shape[1]
         
        hist = numpy.zeros((d, d, n))
    
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                # Calculate sample's histogram array coords rotated relative to ori.
                # Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
                # r_rot = 1.5) have full weight placed in row 1 after interpolation.
                x_rot = i * cos_t + j * sin_t
                y_rot = j * cos_t - i * sin_t
                xbin = x_rot + d / 2 - 0.5
                ybin = y_rot + d / 2 - 0.5
                xt = x + i
                yt = y + j
    
                if xbin >= 0 and xbin < d - 1 and ybin >= 0 and ybin < d - 1 and xt > 0 and xt < width - 1 and yt > 0 and yt < height - 1:
                    dx = I[x+1, y] - I[x-1, y]
                    dy = I[x, y+1] - I[x, y-1]
                    grad_angle = numpy.arctan2(dy, dx)
                    grad_mag = numpy.sqrt(dx * dx + dy * dy) * numpy.exp((x_rot * x_rot + y_rot * y_rot) * exp_scale)
                    obin = (grad_angle - angle) * bins_per_rad
                    x0 = int(numpy.floor(xbin))
                    y0 = int(numpy.floor(ybin))
                    o0 = int(numpy.floor(obin))
                    xbin -= x0
                    ybin -= y0
                    obin -= o0
                    if o0 < 0:
                        o0 += n
                    if o0 >= n:
                        o0 -= n
                    
                    # histogram update using tri-linear interpolation
                    v_x1 = grad_mag * xbin
                    v_x0 = grad_mag - v_x1
                    v_xy11 = v_x1 * ybin
                    v_xy10 = v_x1 - v_xy11
                    v_xy01 = v_x0 * ybin
                    v_xy00 = v_x0 - v_xy01
                    v_xyo111 = v_xy11 * obin
                    v_xyo110 = v_xy11 - v_xyo111
                    v_xyo101 = v_xy10 * obin
                    v_xyo100 = v_xy10 - v_xyo101
                    v_xyo011 = v_xy01 * obin
                    v_xyo010 = v_xy01 - v_xyo011
                    v_xyo001 = v_xy00 * obin
                    v_xyo000 = v_xy00 - v_xyo001
                    
                    hist[x0, y0, o0] += v_xyo000
                    hist[x0, y0, (o0 + 1) % n] += v_xyo001
                    hist[x0, y0 + 1, o0] += v_xyo010
                    hist[x0, y0 + 1, (o0 + 1) % n] += v_xyo011
                    hist[x0 + 1, y0, o0] += v_xyo100
                    hist[x0 + 1, y0, (o0 + 1) % n] += v_xyo101
                    hist[x0 + 1, y0 + 1, o0] += v_xyo110
                    hist[x0 + 1, y0 + 1, (o0 + 1) % n] += v_xyo111

        # copy histogram to the descriptor,
        # apply hysteresis thresholding
        # and scale the result, so that it can be easily converted
        # to byte array
        hist = hist.flatten()
        
        nrm2 = 0
        for i in range(len(hist)):
            nrm2 += hist[i] ** 2
        threshold = numpy.sqrt(nrm2) * SIFT_DESCR_MAG_THR
        
        nrm2 = 0
        for i in range(len(hist)):
            val = numpy.min(hist[i], threshold)
            hist[i] = val
            nrm2 += val * val
        factor = 1 / numpy.max(numpy.sqrt(nrm2), 0.0000001)
    
        for i in range(len(hist)):
            hist[i] *= factor
        return hist
    
    def _calc_descriptors(self, unflatten):
        ret = []
        for i in range(len(self.keypoints)):
            kpt = self.keypoints[i]
            assert kpt.octave >= -1 and kpt.layer <= self.noctave_layers + 2
            scale = 1 / numpy.exp2(kpt.octave)
            size = kpt.sigma * scale
            img = self.gaussian_pyramid[(kpt.octave + 1) * (self.noctave_layers + 3) + int(numpy.round(kpt.layer))]
            ret.append(self._calc_SIFT_descriptor(img, kpt.x * scale, kpt.y * scale, kpt.angle, size))
        if unflatten:
            return numpy.array(ret)
        return numpy.array(ret).flatten()
    
    def calc_features_for_image(self, I, unflatten):
        self.noctaves = int(numpy.round(numpy.log2(min(I.shape[0], I.shape[1])))) - 1
        self._create_initial_image(I)
        self._build_gaussian_pyramid()
        self._build_DoG_pyramid()
        self._find_scale_space_extrema()
        
        assert len(self.keypoints) > 0
        self.keypoints.sort(key=lambda kpt: kpt.response, reverse=True)
        # remove duplicate
        filtered_keypoints = [self.keypoints[0]]
        for i in range(1, len(self.keypoints)):
            if self.keypoints[i].x != self.keypoints[i - 1].x or \
                    self.keypoints[i].y != self.keypoints[i - 1].y or \
                    self.keypoints[i].sigma != self.keypoints[i - 1].sigma or \
                    self.keypoints[i].angle != self.keypoints[i - 1].angle:
                filtered_keypoints.append(self.keypoints[i])
        # retain best
        if len(self.keypoints) > self.nfeatures:
            self.keypoints = filtered_keypoints[:self.nfeatures]
        elif not unflatten:
            for i in range(len(self.keypoints), self.nfeatures):
                self.keypoints.append(self.keypoints[0].clone())
        
        for kpt in self.keypoints:
            kpt.octave -= 1
            kpt.x /= 2
            kpt.y /= 2
            kpt.sigma /= 2
            
        if plot:  
            _, ax = plt.subplots(1)
            ax.set_aspect('equal')  
            ax.imshow(I * 2.5 + 0.5, interpolation='none')
            for kpt in self.keypoints:
                circle = Circle((kpt.y, kpt.x), kpt.sigma)
                ax.add_patch(circle)
            
            plt.show()
        
        return self._calc_descriptors(unflatten)
    