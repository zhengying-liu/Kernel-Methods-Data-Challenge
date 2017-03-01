import numpy

class HOGFeatureExtractor:
    """
    xstep: step on the horizontal direction when displacing the center
    ystep: step on the vertical direction when displacing the center
    """
    def __init__(self, xstep, ystep):
        assert xstep > 0 and ystep > 0
        self.xstep = xstep
        self.ystep = ystep

    def _calc_gradient_for_channel(self, I):
        nX, nY = I.shape
        histogram = [0] * 9

        for i in range(1, nX - 1, self.xstep):
            for j in range(1, nY - 1, self.ystep):
                dx = I[i + 1, j] - I[i - 1, j]
                dy = I[i, j + 1] - I[i, j - 1]

                magnitude = numpy.sqrt(dx**2 + dy**2)
                angle = numpy.arctan2(dx, dy)
                angle = (angle + numpy.pi) / (2 * numpy.pi / 9)
                bin_pos = int(numpy.floor(angle))

                # handle corner case
                if bin_pos == 9:
                    bin_pos = 0
                    angle = 0

                closest_bin = bin_pos

                if bin_pos == 0:
                    if angle < 0.5:
                        second_closest_bin = 8
                    else:
                        second_closest_bin = 1
                elif bin_pos == 8:
                    if angle < 8.5:
                        second_closest_bin = 7
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
                histogram[closest_bin] += r * magnitude
                histogram[second_closest_bin] += (1 - r) * magnitude

        return histogram

    def _calc_gradient_for_image(self, I):
        nchannels = I.shape[2]
        ret = []

        for i in range(nchannels):
            ret += self._calc_gradient_for_channel(I[:,:,i])

        return ret

    def predict(self, X):
        assert X.ndim == 4
        print("Extracting HOG features")
        n = X.shape[0]
        ret = []

        for i in range(n):
            ret.append(self._calc_gradient_for_image(X[i,:,:,:]))

        return numpy.array(ret)
