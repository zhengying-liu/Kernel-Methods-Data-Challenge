"""
Classification into multiple classes with cross entropy loss
"""

import numpy

class CrossEntropyClassifier:
    """
    nclasses: number of classes (assumed between 0 and nclasses - 1)
    W: projection matrix of size (dimension of data) x nclasses
    """

    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.W = None

    def _calc_loss(self, X, y):
        n = X.shape[0]
        loss = 0.0

        P = numpy.dot(X, self.W)
        P = numpy.exp(P)
        sumP = numpy.sum(P, axis=1)

        for i in range(n):
            loss += -numpy.log(P[i, y[i]] / sumP[i])

        return loss / n

    def _calc_gradient(self, X, y):
        n,d = X.shape

        P = numpy.dot(X, self.W)
        P = numpy.exp(P)
        sumP = numpy.sum(P, axis=1)

        gradW = numpy.zeros(self.W.shape)
        for i in range(n):
            P[i,:] = P[i,:] / sumP[i]
            for j in range(self.nclasses):
                c = P[i,j] / P[i, y[i]]
                if j == y[i]:
                    c -= 1 / P[i, y[i]]
                gradW[:, j] += c * X[i, :].T

        return gradW

    """
    X: matrix of size (number of data samples) x (dimension of data)
    iterations: number of gradient descent steps
    lr: fixed learning rate
    """
    def fit(self, X, y, iterations=10, lr=0.001):
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1
        assert iterations > 0

        n, d = X.shape

        self.W = numpy.zeros((d, self.nclasses))
        history = {
            'loss': [self._calc_loss(X, y)]
        }

        for it in range(iterations):
            gradW = self._calc_gradient(X, y)
            self.W -= lr * gradW
            history['loss'].append(self._calc_loss(X, y))

        return history

    def predict(self, X):
        # TODO
        pass

if __name__ == '__main__':
    model = CrossEntropyClassifier(2)
    X = numpy.array([[1,0], [-1,0]])
    y = numpy.array([0,1])
    history = model.fit(X,y)
    print(history)
