"""
Classification into multiple classes with cross entropy loss
"""

from tqdm import tqdm
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
        P -= numpy.max(P)
        P = numpy.exp(P)
        sumP = numpy.sum(P, axis=1)

        for i in range(n):
            p = P[i, y[i]] / sumP[i]
            loss += -numpy.log(p)

        return loss / n

    def _calc_gradient(self, X, y):
        n = X.shape[0]

        P = numpy.dot(X, self.W)
        P -= numpy.max(P)
        P = numpy.exp(P)
        sumP = numpy.sum(P, axis=1)

        gradW = numpy.zeros(self.W.shape)
        for i in range(n):
            P[i,:] = P[i,:] / sumP[i]
            for j in range(self.nclasses):
                c = P[i,j] / P[i, y[i]]
                if j == y[i]:
                    c -= 1 / P[i, y[i]]
                gradW[:, j] += c * X[i, :]

        return gradW / n

    """
    X: matrix of size (number of data samples) x (dimension of data)
    iterations: number of gradient descent steps
    lr: fixed learning rate
    validation: ratio of data that will be used for cross-validation (between 0 and 1)
    """
    def fit(self, X, y, iterations=10, lr=0.001, validation=None, early_stopping=None):
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1
        assert iterations > 0

        n, d = X.shape

        self.W = numpy.zeros((d, self.nclasses))
        history = {}

        if validation is not None:
            assert validation > 0 and validation < 1
            split_idx = int(validation * n)
            Xval = X[split_idx:,:]
            yval = y[split_idx:]
            X = X[:split_idx,:]
            y = y[:split_idx]
            history['val_loss'] = [self._calc_loss(Xval, yval)]
            history['val_accuracy'] = [self._calc_accuracy(Xval, yval)]
            best_validation_index = 0

        history['loss'] = [self._calc_loss(X, y)]
        history['accuracy'] = [self._calc_accuracy(X, y)]

        for it in tqdm(range(iterations)):
            gradW = self._calc_gradient(X, y)
            self.W -= lr * gradW
            history['loss'].append(self._calc_loss(X, y))
            history['accuracy'].append(self._calc_accuracy(X, y))

            if validation is not None:
                history['val_loss'].append(self._calc_loss(Xval, yval))
                history['val_accuracy'].append(self._calc_accuracy(Xval, yval))

                if history['val_accuracy'][-1] > history['val_accuracy'][best_validation_index]:
                    best_validation_index = it
                if it + 1 == best_validation_index + early_stopping:
                    break

        return history

    def predict(self, X, probability=False):
        n = X.shape[0]
        P = numpy.dot(X, self.W)

        if probability:
            P -= numpy.max(P)
            P = numpy.exp(P)
            sumP = numpy.sum(P, axis=1)

            for i in range(n):
                P[i, :] = P[i, :] / sumP[i]

            return P
        else:
            y = numpy.argmax(P, axis=1)
            return y

    def _calc_accuracy(self, X, y):
        ypred = self.predict(X)
        return numpy.sum(ypred == y) * 100.0 / y.shape[0]

if __name__ == '__main__':
    model = CrossEntropyClassifier(2)
    X = numpy.array([[1,0], [-1,0]])
    y = numpy.array([0,1])
    history = model.fit(X,y)
    print(history)
    print(model.predict(X))
    print(model.predict(X, probability=True))
