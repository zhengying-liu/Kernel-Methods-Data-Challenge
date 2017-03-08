class Keypoint:
    """
    response: A score to tell how important the keypoint is.
        It is the norm of the second derivative of the adjusted local extrema.
    """
    def __init__(self, nclusters):
        self.x = 0.
        self.y = 0.
        self.octave = 0
        self.layer = 0.
        self.sigma = 0.
        self.angle = 0.
        self.response = 0
