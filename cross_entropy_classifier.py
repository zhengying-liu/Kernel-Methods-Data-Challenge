"""
Classification into multiple classes with cross entropy loss
"""

import numpy

class CrossEntropyClassifier:
    """
    nclasses: number of classes (assumed between 0 and nclasses - 1)
    W: projection matrix of size dimension of data x nclasses
    """

    def __init__(self, nclasses):
        self.nclasses = nclasses

    def _calc_loss(X, y):
        n = X.shape[0]
        loss = 0.0

        for i in range(0,n):
            p = numpy.dot(X[i, :], self.W[:, y[i]])
            loss += -numpy.log(p)

        return loss / N

    def _calc_gradient(X, y1hot):
        # TODO
        pass

    """
    iterations: number of gradient descent steps
    lr: fixed learning rate
    """
    def fit(self, X, y, iterations=10, lr=0.001):
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1
        assert iterations > 0

        n, d = X.shape

        y1hot = numpy.zeros((n, self.nclasses))
        for i in range(n):
            y1hot[i, y[i]] = 1

        self.W = numpy.zeros((d, self.nclasses))
        history = {
            'loss': [_calc_loss(X, y)]
        }

        for it in range(iterations):
            grad = _calc_gradient(X, y2)
            self.W -= lr * grad
            history['loss'].append(_calc_loss(X, y))

        return history

    def predict(self, X):
        # TODO
        pass
