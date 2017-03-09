import numpy

def pca(X, components):
    assert components > 0 and X.shape[1] >= components
    U, s, Vt = numpy.linalg.svd(numpy.matrix(X), full_matrices=False)
    V = Vt.T
    S = numpy.diag(s)
    return numpy.array(numpy.dot(U[:, :components], S[:components, :components])), V[:, :components]
