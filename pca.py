import numpy

def pca(X, components):
    assert components > 0 and X.shape[1] >= components
    U, s, _ = numpy.linalg.svd(numpy.matrix(X), full_matrices=False)
    S = numpy.diag(s)
    return numpy.array(numpy.dot(U[:, :components], S[:components, :components]))
