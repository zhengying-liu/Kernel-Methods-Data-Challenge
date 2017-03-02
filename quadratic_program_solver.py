import numpy

class QuadraticProgramSolver:
    def _f(self, Q, p, A, b, x, t):
        B = b - numpy.dot(A, x)

        if numpy.sum(B <= 0) > 0:
            return numpy.inf
        else:
            return t * (numpy.dot(numpy.dot(x, Q), x) / 2 + numpy.dot(p, x)) - numpy.sum(numpy.log(B))

    def _gradf(self, Q, p, A, b, x, t):
        c = 1.0 / (b - numpy.dot(A, x))
        return t * (numpy.dot(Q, x) + p) + numpy.dot(A.T, c)

    def _hessianf(self, Q, A, b, x, t):
        d = numpy.diag(1.0 / (b - numpy.dot(A, x))**2)
        return t * Q + numpy.dot(numpy.dot(A.T, d), A)

    def _newton(self, Q, p, A, b, x, t):
        fxt = self._f(Q, p, A, b, x, t)
        gxt = self._gradf(Q, p, A, b, x, t)
        Hxt = self._hessianf(Q, A, b, x, t)
        invHxt = numpy.linalg.inv(Hxt)
        delta = -numpy.dot(invHxt, gxt)

        rho = 1
        alpha = 0.25
        beta = 0.5

        while self._f(Q, p, A, b, x + rho * delta, t) > fxt + alpha * rho * numpy.dot(gxt, delta):
            rho = beta * rho

        x = x + rho * delta
        gap = numpy.dot(numpy.dot(gxt, invHxt), gxt) / 2
        assert gap >= 0
        return x, gap

    def _centering_step(self, Q, p, A, b, x, t, tol):
        error = tol + 1

        while error >= tol:
            x, error = self._newton(Q, p, A, b, x, t)

        return x

    def barrier_method(self, Q, p, A, b, x0, mu, tol):
        assert x0.ndim == 1
        #print("Starting quadratic program solver")
        x = x0
        error = tol + 1
        m = float(A.shape[0])
        t = 1.0

        if A.ndim == 1:
            A = A[numpy.newaxis, :]

        while error >= tol:
            x = self._centering_step(Q, p, A, b, x, t, tol)
            error = m / t
            t = t * mu

        return x

if __name__ == '__main__':
    solver = QuadraticProgramSolver()
    Q = numpy.array([[2,0],[0,2]])
    p = numpy.array([0,0])
    A = numpy.array([1,1])
    b = numpy.array([1])
    x0 = numpy.array([-1, -1])

    x = solver.barrier_method(Q, p, A, b, x0, 2, 1e-8)
    print x
