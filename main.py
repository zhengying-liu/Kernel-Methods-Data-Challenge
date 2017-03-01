import matplotlib.pyplot as plt
import numpy

from cross_entropy_classifier import CrossEntropyClassifier
from hog_feature_extractor import HOGFeatureExtractor
from kernel_pca import KernelPCA
from kernels import LinearKernel, GaussianKernel
from utils import load_data, plot_history, write_output, concat_bias

output_suffix = 'trial6'
feature_extractor = 'hog'

print("Loading data")
Xtrain, Ytrain, Xtest = load_data()

if feature_extractor == 'hog':
    hog = HOGFeatureExtractor(1, 1)
    Xtrain = hog.predict(Xtrain)
    Xtest = hog.predict(Xtest)

Xtrain = numpy.reshape(Xtrain, (Xtrain.shape[0], -1))
Xtest = numpy.reshape(Xtest, (Xtest.shape[0], -1))

kernel_pca = False

if kernel_pca:
    print("Kernel PCA")
    pca = KernelPCA(GaussianKernel(0.1))
    X = numpy.concatenate((Xtrain, Xtest), axis=0)
    pca.fit(X)

    components = min(1000, Xtrain.shape[1])
    X = pca.predict(components)
    ntrain = Xtrain.shape[0]
    Xtrain = X[:ntrain, :]
    Xtest = X[ntrain:, :]

Xtrain = concat_bias(Xtrain)
Xtest = concat_bias(Xtest)

print("Fitting on training data")
model = CrossEntropyClassifier(10)
iterations = 500
lr = 0.01
history = model.fit(Xtrain, Ytrain, iterations, lr, 0.2, 10)

best = numpy.argmax(history['val_accuracy'])
print("Best accuracy is %.3f at iteration %d" % (history['val_accuracy'][best], best))

f = plot_history(history)
f.savefig('plots/' + output_suffix + '.png')

model = CrossEntropyClassifier(10)
history = model.fit(Xtrain, Ytrain, best, lr)

print("Predicting on test data")
Ytest = model.predict(Xtest)
write_output(Ytest, 'results/Yte_' + output_suffix + '.csv')
