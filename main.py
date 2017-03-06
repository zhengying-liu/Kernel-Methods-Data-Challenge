import matplotlib.pyplot as plt
import numpy
import os

from cross_entropy_classifier import CrossEntropyClassifier
from fisher_feature_extractor import FisherFeatureExtractor
from hog_feature_extractor import HOGFeatureExtractor
from kernel_descriptors_extractor import KernelDescriptorsExtractor
from kernel_pca import KernelPCA
from kernels import (LinearKernel, GaussianKernel, HistogramIntersectionKernel,
                    LaplacianRBFKernel, SublinearRBFKernel)
from svm import KernelSVMOneVsOneClassifier, KernelSVMOneVsAllClassifier
from utils import load_data, plot_history, write_output, concat_bias

output_suffix = 'trial14'
feature_extractor = 'hog'
classifier = 'svm_ovo'
validation = 0.2
nclasses = 10
overwrite = False
kernel_pca = False

print("Loading data")
Xtrain, Ytrain, Xtest = load_data()

if feature_extractor == 'hog':
    hog = HOGFeatureExtractor()

    if not overwrite and os.path.isfile('data/Xtrain_hog.npy'):
        Xtrain = numpy.load('data/Xtrain_hog.npy')
    else:
        Xtrain = hog.predict(Xtrain)
        numpy.save('data/Xtrain_hog', Xtrain)

    if not overwrite and os.path.isfile('data/Xtest_hog.npy'):
        Xtest = numpy.load('data/Xtest_hog.npy')
    else:
        Xtest = hog.predict(Xtest)
        numpy.save('data/Xtest_hog', Xtest)
elif feature_extractor == 'hog_fisher':
    fisher = FisherFeatureExtractor(nclasses=5)
    
    if not overwrite and os.path.isfile('data/Xtrain_hog_fisher.npy'):
        Xtrain = numpy.load('data/Xtrain_hog_fisher.npy')
    else:
        Xtrain = fisher.predict(Xtrain)
        numpy.save('data/Xtrain_hog_fisher', Xtrain)

    if not overwrite and os.path.isfile('data/Xtest_hog_fisher.npy'):
        Xtest = numpy.load('data/Xtest_hog_fisher.npy')
    else:
        Xtest = fisher.predict(Xtest)
        numpy.save('data/Xtest_hog_fisher', Xtest)
elif feature_extractor == 'kernel_descriptors':
    kdes = KernelDescriptorsExtractor()

    if not overwrite and os.path.isfile('data/Xtrain_kdes.npy'):
        Xtrain = numpy.load('data/Xtrain_kdes.npy')
    else:
        Xtrain = kdes.predict(Xtrain)
        numpy.save('data/Xtrain_kdes.npy', Xtrain)

    if not overwrite and os.path.isfile('data/Xtest_kdes.npy'):
        Xtest = numpy.load('data/Xtest_kdes.npy')
    else:
        Xtest = kdes.predict(Xtest)
        numpy.save('data/Xtest_kdes.npy', Xtest)
elif feature_extractor == 'raw':
    pass
else:
    raise Exception("Unknown feature extractor")

Xtrain = numpy.reshape(Xtrain, (Xtrain.shape[0], -1))
Xtest = numpy.reshape(Xtest, (Xtest.shape[0], -1))

print(Xtrain.shape)
print(Xtest.shape)

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
    print(Xtrain.shape)
    print(Xtest.shape)

print("Fitting on training data")
if classifier == 'cross_entropy':
    Xtrain = concat_bias(Xtrain)
    Xtest = concat_bias(Xtest)

    model = CrossEntropyClassifier(nclasses)
    iterations = 500
    lr = 0.01
    history = model.fit(Xtrain, Ytrain, iterations, lr, validation, 10)

    best = numpy.argmax(history['val_accuracy'])
    print("Best accuracy is %.3f at iteration %d" % (history['val_accuracy'][best], best))

    f = plot_history(history)
    f.savefig('plots/' + output_suffix + '.png')

    model = CrossEntropyClassifier(nclasses)
    history = model.fit(Xtrain, Ytrain, best, lr)
elif classifier == 'svm_ovo':
    #kernel = GaussianKernel(0.6)
    #kernel = HistogramIntersectionKernel(0.25)
    kernel = LaplacianRBFKernel(1.6)
    #kernel = SublinearRBFKernel(0.4)
    #kernel = LinearKernel()
    model = KernelSVMOneVsOneClassifier(nclasses, kernel)
    C = 1
    K = kernel.build_K(Xtrain)
    model.fit(Xtrain, Ytrain, C, validation, K=K)

    model = KernelSVMOneVsOneClassifier(nclasses, kernel)
    model.fit(Xtrain, Ytrain, C, K=K)
elif classifier == 'svm_ova':
    kernel = GaussianKernel(1.5)
    reg_lambda = 0.5
    model = KernelSVMOneVsAllClassifier(nclasses, kernel)
    model.fit(Xtrain, Ytrain, reg_lambda, validation)

    model = KernelSVMOneVsAllClassifier(nclasses, kernel)
    model.fit(Xtrain, Ytrain, reg_lambda)
else:
    raise Exception("Unknown classifier")

print("Predicting on test data")
Ytest = model.predict(Xtest)
write_output(Ytest, 'results/Yte_' + output_suffix + '.csv')
