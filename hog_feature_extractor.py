from tqdm import tqdm
import numpy

# References:
# http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/
# http://www.learnopencv.com/histogram-of-oriented-gradients/
class HOGFeatureExtractor:
    """
    nbins: number of bins that will be used
    unsigned: if True the sign of the angle is not considered
    """
    def __init__(self, nbins=9, unsigned=True):
        self.nbins = nbins
        self.unsigned = unsigned

    def _calc_gradient_for_channel(self, I):
        nX, nY = I.shape
        histogram = numpy.zeros((4, 4, self.nbins))

        for i in range(0, nX):
            for j in range(0, nY):
                dx, dy = 0, 0
                if i < nX - 1:
                    dx += I[i + 1, j]
                if i > 0:
                    dx -= I[i - 1, j]
                if j < nY - 1:
                    dy += I[i, j + 1]
                if j > 0:
                    dy -= I[i, j - 1]

                if dy == 0 and dx == 0:
                    continue

                magnitude = numpy.sqrt(dx**2 + dy**2)
                if self.unsigned:
                    if dx == 0:
                        angle = numpy.pi / 2
                    else:
                        angle = numpy.arctan(dy / dx)
                    angle = (angle + numpy.pi / 2) / (numpy.pi / self.nbins)
                else:
                    angle = numpy.arctan2(dx, dy)
                    angle = (angle + numpy.pi) / (2 * numpy.pi / self.nbins)

                bin_pos = int(numpy.floor(angle))
                # handle corner case
                if bin_pos == self.nbins:
                    bin_pos = 0
                    angle = 0

                closest_bin = bin_pos

                if bin_pos == 0:
                    if angle < 0.5:
                        second_closest_bin = self.nbins - 1
                    else:
                        second_closest_bin = 1
                elif bin_pos == self.nbins - 1:
                    if angle < self.nbins - 0.5:
                        second_closest_bin = self.nbins - 2
                    else:
                        second_closest_bin = 0
                else:
                    if angle < bin_pos + 0.5:
                        second_closest_bin = bin_pos - 1
                    else:
                        second_closest_bin = bin_pos + 1

                # closest_bin_distance + second_closest_bin_distance = 1
                if angle < bin_pos + 0.5:
                    second_closest_bin_distance = angle - (bin_pos - 0.5)
                else:
                    second_closest_bin_distance = (bin_pos + 1.5) - angle

                r = second_closest_bin_distance
                histogram[i / 8, j / 8, closest_bin] += r * magnitude
                histogram[i / 8, j / 8, second_closest_bin] += (1 - r) * magnitude

        ret = numpy.zeros((3, 3, self.nbins * 4))

        for i in range(3):
            for j in range(3):
                aux = histogram[i:i + 2, j:j + 2, :].flatten().copy()
                aux = aux / numpy.linalg.norm(aux)
                ret[i, j, :] = aux

        return ret.flatten()

    def _calc_gradient_for_image(self, I):
        nchannels = I.shape[2]
        ret = []

        for i in range(nchannels):
            ret.append(self._calc_gradient_for_channel(I[:,:,i]))

        return numpy.array(ret).flatten()

    def predict(self, X):
        assert X.ndim == 4
        print("Extracting HOG features")
        n = X.shape[0]
        ret = []

        for i in tqdm(range(n)):
            ret.append(self._calc_gradient_for_image(X[i,:,:,:]))

        return numpy.array(ret)
