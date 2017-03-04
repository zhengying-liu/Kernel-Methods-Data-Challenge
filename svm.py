from tqdm import tqdm
import numpy
import random
random.seed(123)

from kernels import build_K
from quadratic_program_solver import QuadraticProgramSolver
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

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
        Q = matrix(Q, tc='d')

        p = numpy.zeros(2 * n)
        p[n:] = 1.0 / (2 * reg_lambda * n)
        p = matrix(p, tc='d')

        A = numpy.zeros((2 * n, 2 * n))
        A[:n, :n] = -(K * y).T
        A[:n, n:] = -numpy.eye(n)
        A[n:, n:] = -numpy.eye(n)
        A = matrix(A, tc='d')

        b = numpy.zeros(2 * n)
        b[:n] = -1
        b = matrix(b, tc='d')

        return Q, p, A, b

    # Quadractic programming
    def _solve_primal(self, y, K, reg_lambda):
        n = K.shape[0]
        Q, p, A, b = self._svm_primal(y, K, reg_lambda)

        w0 = numpy.zeros(2 * n)
        w0[n:] = 2
        #solver = QuadraticProgramSolver()
        #w = solver.barrier_method(Q, p, A, b, w0, 2, 1e-5)
        w = solvers.qp(Q, p, A, b)['x']
        w = numpy.array(w)[:,0]

        alpha = w[:n]
        return alpha

    # SMO algorithm
    def _solve_dual(self, y, K, reg_lambda, iterations=100):
        #print("Starting SMO to solve dual")
        n = y.shape[0]
        alpha = numpy.zeros(n)

        #result = 2 * numpy.dot(alpha, y) - numpy.dot(numpy.dot(alpha, K), alpha)
        #print "Initial Result dual: %.5f" % result

        for it in range(iterations):
            alpha_prev = alpha.copy()

            for i in range(0,n):
                #i = random.randint(0,n - 1)
                j = random.randint(0,n - 2)
                if j >= i:
                    j += 1

                s = alpha[i] + alpha[j]
                L = max(-(1 - y[i]) / (4 * reg_lambda * n), s - (1 + y[j]) / (4 * reg_lambda * n))
                H = min((1 + y[i]) / (4 * reg_lambda * n), s + (1 - y[j]) / (4 * reg_lambda * n))

                alpha_new = y[j] - y[i] + s * (K[i, j] - K[j, j])
                for k in range(0,n):
                    if k != i and k != j:
                        alpha_new += (K[i, k] - K[j, k]) * alpha[k]
                alpha_new /= 2 * K[i, j] - K[i, i] - K[j, j]

                if L <= alpha_new and alpha_new <= H:
                    alpha[i] = alpha_new
                elif alpha_new < L:
                    alpha[i] = L
                else:
                    alpha[i] = H
                alpha[j] = s - alpha[i]

            diff = numpy.linalg.norm(alpha - alpha_prev)
            if diff < 1e-4:
                break

            #result = 2 * numpy.dot(alpha, y) - numpy.dot(numpy.dot(alpha, K), alpha)
            #print "Result dual: %.5f, diff = %.5f" % (result, diff)

        #result = 2 * numpy.dot(alpha, y) - numpy.dot(numpy.dot(alpha, K), alpha)
        #print "Final Result dual: %.5f" % result

        return alpha

    def fit(self, X, y, K_function, reg_lambda, K=None):
        #print("Fit KernelSVMBinaryClassifier")
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1

        n = X.shape[0]

        if K is None:
            K = build_K(X, K_function)
        else:
            assert K.ndim == 2 and K.shape[0] == n and K.shape[1] == n

        # change y to -1/1
        self.class1 = numpy.min(y)
        self.class2 = numpy.max(y)
        assert self.class1 != self.class2
        ind1 = (y == self.class1)
        ind2 = (y == self.class2)
        #print "points of class %d : %d" % (self.class1, numpy.sum(ind1))
        #print "points of class %d : %d" % (self.class2, numpy.sum(ind2))
        y2 = numpy.zeros(n)
        y2[ind1] = -1
        y2[ind2] = 1

        reg_lambda = float(reg_lambda)
        #alpha = self._solve_primal(y2, K, reg_lambda)
        #print alpha
        alpha = self._solve_dual(y2, K, reg_lambda)
        #print alpha
        ind = (numpy.abs(alpha) > 1e-9)
        n_support_vectors = numpy.sum(ind)
        #print("support vectors: %d (of %d)" % (numpy.sum(ind), n))
        assert n_support_vectors > 0
        self.X = X[ind, :]
        self.alpha = alpha[ind]
        self.K_function = K_function
        print "Accuracy on training data: %.3f" % self._calc_accuracy(X, y)

    def predict(self, X, confidence=False):
        n = X.shape[0]
        y = numpy.zeros(n, dtype=numpy.int32)

        for i in range(n):
            pred = 0
            x = X[i,:]
            for v, alpha in zip(self.X, self.alpha):
                pred += alpha * self.K_function.calc(v, x)
            if confidence:
                y[i] = pred
            else:
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
            Xval = X[:split_idx,:]
            yval = y[:split_idx]
            Xtrain = X[split_idx:,:]
            ytrain = y[split_idx:]
        else:
            Xtrain = X
            ytrain = y

        ind_by_class = []
        for i in range(self.nclasses):
            ind = (ytrain == i)
            ind_by_class.append(ind)

        K = build_K(Xtrain, K_function)

        pbar = tqdm(total=self.nclasses * (self.nclasses - 1) / 2)
        for i in range(self.nclasses):
            for j in range(i + 1, self.nclasses):
                ind = numpy.logical_or(ind_by_class[i], ind_by_class[j])
                partial_K = K[ind, :]
                partial_K = partial_K[:, ind]
                self.SVMMatrix[i][j - i - 1].fit(Xtrain[ind, :], ytrain[ind], K_function, reg_lambda, K=partial_K)
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

class KernelSVMOneVsAllClassifier:
    """
    nclasses: number of classes (assumed between 0 and nclasses - 1)
    SVMova: list of one vs all classifiers
    """
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.SVMova = []

        for i in range(self.nclasses):
            self.SVMova.append(KernelSVMBinaryClassifier())

    def fit(self, X, y, K_function, reg_lambda, validation=None):
        print("Fit KernelSVMOneVsAllClassifier")
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1

        n = X.shape[0]

        if validation is not None:
            assert validation > 0 and validation < 1
            split_idx = int(validation * n)
            Xval = X[:split_idx,:]
            yval = y[:split_idx]
            Xtrain = X[split_idx:,:]
            ytrain = y[split_idx:]
        else:
            Xtrain = X
            ytrain = y

        K = build_K(Xtrain, K_function)

        for i in tqdm(range(self.nclasses)):
            y2 = -numpy.ones(ytrain.shape[0])
            ind = (ytrain == i)
            y2[ind] = 1
            self.SVMova[i].fit(Xtrain, y2, K_function, reg_lambda, K=K)

        if validation is not None:
            accuracy = self._calc_accuracy(Xval, yval)
            print("Accuracy in validation data is %.3f" % accuracy)

    def predict(self, X):
        n = X.shape[0]
        scores = numpy.zeros((n, self.nclasses))

        print("One vs All prediction")
        for i in tqdm(range(self.nclasses)):
            scores[:, i] = self.SVMova[i].predict(X, confidence=True)

        return numpy.argmax(scores, axis=1)

    def _calc_accuracy(self, X, y):
        ypred = self.predict(X)
        return numpy.sum(ypred == y) * 100.0 / y.shape[0]

if __name__ == '__main__':
    from kernels import GaussianKernel
    #X = numpy.array([[3,4],[1,3],[2,2]])
    X = numpy.array([[-2,0],[-1,0],[1,0]])
    y = numpy.array([-1,-1,1])

    model = KernelSVMBinaryClassifier()
    model.fit(X, y, GaussianKernel(0.5), 0.5)
    print model.X
    print model.alpha
