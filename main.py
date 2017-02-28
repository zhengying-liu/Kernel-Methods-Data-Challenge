import matplotlib.pyplot as plt
import numpy

from cross_entropy_classifier import CrossEntropyClassifier
from kernel_pca import KernelPCA
from kernels import LinearKernel, GaussianKernel
from utils import load_data, plot_history, write_output

output_suffix = 'trial1'

print("Loading data")
Xtrain, Ytrain, Xtest = load_data()
Xtrain = numpy.reshape(Xtrain, (Xtrain.shape[0], -1))
Xtest = numpy.reshape(Xtest, (Xtest.shape[0], -1))

kernel_pca = False

if kernel_pca:
    print("Kernel PCA")
    pca = KernelPCA(LinearKernel())
    X = numpy.concatenate((Xtrain, Xtest), axis=0)
    pca.fit(X)

    components = 1000
    X = pca.predict(components)
    ntrain = Xtrain.shape[0]
    Xtrain = X[:ntrain, :]
    Xtest = X[ntrain:, :]

print("Fitting on training data")
model = CrossEntropyClassifier(10)
iterations = 40
history = model.fit(Xtrain, Ytrain, iterations, 0.1, 0.2, 10)

best = numpy.argmax(history['val_accuracy'])
print("Best accuracy is %.3f at iteration %d" % (history['val_accuracy'][best], best))

f = plot_history(history)
f.savefig('plots/' + output_suffix + '.png')

model = CrossEntropyClassifier(10)
history = model.fit(Xtrain, Ytrain, best, 0.1)

print("Predicting on test data")
Ytest = model.predict(Xtest)
write_output(Ytest, 'results/Yte_' + output_suffix + '.csv')
