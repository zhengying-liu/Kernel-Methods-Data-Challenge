from tqdm import tqdm
import numpy

from kernels import build_K
from quadratic_program_solver import QuadraticProgramSolver

class KernelSVMBinaryClassifier:
    """
    X: support vectors
    alpha: corresponding coefficients for predictions
    K_function: SVM's kernel
    class1, class2: original labels of the classes
    """
    def __init__(self):
        self.X = None
        self.alpha = None
        self.K_function = None
        self.class1 = None
        self.class2 = None

    def _svm_primal(self, y, K, reg_lambda):
        n = K.shape[0]

        Q = numpy.zeros((2 * n, 2 * n))
        Q[:n, :n] = K

        p = numpy.zeros(2 * n)
        p[n:] = 1.0 / (2 * reg_lambda * n)

        A = numpy.zeros((2 * n, 2 * n))
        A[:n, :n] = -(K * y).T
        A[:n, n:] = -numpy.eye(n)
        A[n:, n:] = -numpy.eye(n)

        b = numpy.zeros(2 * n)
        b[:n] = -1

        return Q, p, A, b

    def _solve_primal(self, y, K, reg_lambda):
        n = K.shape[0]
        Q, p, A, b = self._svm_primal(y, K, reg_lambda)

        w0 = numpy.zeros(2 * n)
        w0[n:] = 2
        solver = QuadraticProgramSolver()
        w = solver.barrier_method(Q, p, A, b, w0, 2, 1e-5)

        alpha = w[:n]
        return alpha

    def fit(self, X, y, K_function, reg_lambda, K=None):
        #print("Fit KernelSVMBinaryClassifier")
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1

        n = X.shape[0]

        if K is None:
            K = build_K(X, K_function)
        else:
            assert K.ndim == 2 and K.shape[0] == K.shape[1]

        # change y to -1/1
        self.class1 = numpy.min(y)
        self.class2 = numpy.max(y)
        assert self.class1 != self.class2
        ind1 = (y == self.class1)
        ind2 = (y == self.class2)
        y2 = numpy.zeros(n)
        y2[ind1] = -1
        y2[ind2] = 1

        alpha = self._solve_primal(y2, K, reg_lambda)
        ind = (alpha != 0)
        n_support_vectors = numpy.sum(ind)
        #print("support vectors: %d" % numpy.sum(ind))
        assert n_support_vectors > 0

        self.X = X[ind, :]
        self.alpha = alpha[ind]
        self.K_function = K_function
        print "Accuracy on training data: %.3f" % self._calc_accuracy(X, y)

    def predict(self, X):
        n = X.shape[0]
        y = numpy.zeros(n, dtype=numpy.int32)

        for i in range(n):
            pred = 0
            x = X[i,:]
            for v, alpha in zip(self.X, self.alpha):
                pred += alpha * self.K_function.calc(v, x)
            if pred >= 0:
                y[i] = self.class2
            else:
                y[i] = self.class1

        return y

    def _calc_accuracy(self, X, y):
        ypred = self.predict(X)
        return numpy.sum(ypred == y) * 100.0 / y.shape[0]

class KernelSVMOneVsOneClassifier:
    """
    nclasses: number of classes (assumed between 0 and nclasses - 1)
    SVMMatrix: matrix of one vs one classifiers
    """
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.SVMMatrix = []

        for i in range(self.nclasses):
            aux = []
            for j in range(i + 1, self.nclasses):
                aux.append(KernelSVMBinaryClassifier())
            self.SVMMatrix.append(aux)

    def fit(self, X, y, K_function, reg_lambda, validation=None):
        print("Fit KernelSVMOneVsOneClassifier")
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1

        n = X.shape[0]

        if validation is not None:
            assert validation > 0 and validation < 1
            split_idx = int(validation * n)
            Xval = X[split_idx:,:]
            yval = y[split_idx:]
            X = X[:split_idx,:]
            y = y[:split_idx]

        ind_by_class = []
        for i in range(self.nclasses):
            ind = (y == i)
            ind_by_class.append(ind)

        K = build_K(X, K_function)

        pbar = tqdm(total=self.nclasses * (self.nclasses - 1) / 2)
        for i in range(self.nclasses):
            for j in range(i + 1, self.nclasses):
                ind = numpy.logical_or(ind_by_class[i], ind_by_class[j])
                partial_K = K[ind, :]
                partial_K = partial_K[:, ind]
                self.SVMMatrix[i][j - i - 1].fit(X[ind, :], y[ind], K_function, reg_lambda, K=partial_K)
                pbar.update(1)
        pbar.close()

        if validation is not None:
            accuracy = self._calc_accuracy(Xval, yval)
            print("Accuracy in validation data is %.3f" % accuracy)

    def predict(self, X):
        n = X.shape[0]
        scores = numpy.zeros((n, self.nclasses))

        print("One vs One prediction")
        pbar = tqdm(total=self.nclasses * (self.nclasses - 1) / 2)
        for i in range(self.nclasses):
            for j in range(i + 1, self.nclasses):
                y = self.SVMMatrix[i][j - i - 1].predict(X)

                for k, pred in enumerate(y):
                    scores[k][pred] += 1

                pbar.update(1)
        pbar.close()

        return numpy.argmax(scores, axis=1)

    def _calc_accuracy(self, X, y):
        ypred = self.predict(X)
        return numpy.sum(ypred == y) * 100.0 / y.shape[0]
