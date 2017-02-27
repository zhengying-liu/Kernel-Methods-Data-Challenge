import numpy

def reshape_images(X):
    n = X.shape[0]
    X = numpy.reshape(X, (n, 3, 32, 32))
    X = numpy.swapaxes(X, 1, 2)
    X = numpy.swapaxes(X, 2, 3)
    return X

def load_data():
    Xtrain = numpy.genfromtxt('data/Xtr.csv', delimiter=',')
    Xtrain = Xtrain[:,:3072]

    aux = numpy.genfromtxt('data/Ytr.csv', delimiter=',', names=True)
    Ytrain = numpy.zeros((5000,), dtype=numpy.int32)
    for i, y in enumerate(aux):
        Ytrain[i] = int(y[1])

    assert Xtrain.shape[0] == Ytrain.shape[0]

    Xtest = numpy.genfromtxt('data/Xte.csv', delimiter=',')
    Xtest = Xtest[:,:3072]

    assert Xtest.shape[1] == Xtrain.shape[1]

    Xtrain = reshape_images(Xtrain)
    Xtest = reshape_images(Xtest)

    return Xtrain, Ytrain, Xtest

def write_output(Y, filename):
    assert(Y.shape[0] == 2000)
    f = open(filename, 'w')
    f.write('Id,Prediction\n')

    for i, y in enumerate(Y):
        f.write("{0:d},{1:d}\n".format(i + 1, y))

    print("Ytest output to : %s" % filename)

if __name__ == '__main__':
    Xtrain, Ytrain, Xtest = load_data()

    print(Xtrain.shape, Ytrain.shape, Xtest.shape)
    print(Xtrain[0, 0, 0, 0], Xtrain[0, 31, 31, 2])
    print(Ytrain[:10])

    import matplotlib.pyplot as plt
    plt.imshow(Xtrain[0,:,:,:])
    plt.show()
