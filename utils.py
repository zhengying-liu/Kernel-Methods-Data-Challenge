import matplotlib.pyplot as plt
import numpy
import os

n_train = 5000
n_test = 2000
# n_train = 20
# n_test = 10

def reshape_images(X):
    n = X.shape[0]
    X = numpy.reshape(X, (n, 3, 32, 32))
    X = numpy.swapaxes(X, 1, 2)
    X = numpy.swapaxes(X, 2, 3)
    return X

def load_data(folder_name='data/', overwrite=False):
    if not overwrite and os.path.isfile(folder_name + 'Xtrain.npy'):
        Xtrain = numpy.load(folder_name + 'Xtrain.npy')
    else:
        Xtrain = numpy.genfromtxt(folder_name + 'Xtr.csv', delimiter=',')
        Xtrain = Xtrain[:,:3072]
        Xtrain = reshape_images(Xtrain)
        numpy.save(folder_name + 'Xtrain', Xtrain)

    if not overwrite and os.path.isfile(folder_name + 'Ytrain.npy'):
        Ytrain = numpy.load(folder_name + 'Ytrain.npy')
    else:
        aux = numpy.genfromtxt(folder_name + 'Ytr.csv', delimiter=',', names=True)
        Ytrain = numpy.zeros((n_train,), dtype=numpy.int32)
        for i, y in enumerate(aux):
            Ytrain[i] = int(y[1])
        numpy.save(folder_name + 'Ytrain', Ytrain)

    assert Xtrain.shape[0] == Ytrain.shape[0]

    if not overwrite and os.path.isfile(folder_name + 'Xtest.npy'):
        Xtest = numpy.load(folder_name + 'Xtest.npy')
    else:
        Xtest = numpy.genfromtxt(folder_name + 'Xte.csv', delimiter=',')
        Xtest = Xtest[:,:3072]
        Xtest = reshape_images(Xtest)
        numpy.save(folder_name + 'Xtest', Xtest)

    return Xtrain, Ytrain, Xtest

def write_output(Y, filename):
    assert(Y.shape[0] == n_test)
    f = open(filename, 'w')
    f.write('Id,Prediction\n')

    for i, y in enumerate(Y):
        f.write("{0:d},{1:d}\n".format(i + 1, y))

    print("Ytest output to : %s" % filename)

def plot_history(history):
    n = len(history['loss'])
    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(range(n), history['loss'], range(n), history['val_loss'])
    axarr[0].legend(['train', 'validation'], loc='upper left')
    axarr[0].set_ylabel('loss')
    axarr[0].set_xlabel('epoch')

    axarr[1].plot(range(n), history['accuracy'], range(n), history['val_accuracy'])
    axarr[1].legend(['train', 'validation'], loc='upper left')
    axarr[1].set_ylabel('accuracy')
    axarr[1].set_xlabel('epoch')

    return f

def concat_bias(X):
    r, c = X.shape
    aux = numpy.ones((r,c + 1))
    aux[:,:-1] = X
    return aux

if __name__ == '__main__':
    Xtrain, Ytrain, Xtest = load_data()

    print(Xtrain.shape, Ytrain.shape, Xtest.shape)
    print(Xtrain[0, 0, 0, 0], Xtrain[0, 31, 31, 2])
    print(Ytrain[:10])

    import matplotlib.pyplot as plt
    plt.imshow(Xtrain[0,:,:,:])
    plt.show()
