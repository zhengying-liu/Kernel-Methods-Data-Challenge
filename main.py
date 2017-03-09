import numpy

from cross_entropy_classifier import CrossEntropyClassifier
from load_features import load_features
from kernels import (LinearKernel, GaussianKernel, HistogramIntersectionKernel,
                    LaplacianRBFKernel, SublinearRBFKernel)
from svm import KernelSVMOneVsOneClassifier, KernelSVMOneVsAllClassifier
from utils import plot_history, write_output, concat_bias

output_suffix = 'trial15'

feature_extractor = 'sift_fisher'
overwrite_features = False

overwrite_kpca = False
kernel_pca = True
kernel_pca_kernel = GaussianKernel(0.6)
cut_percentage = 90
# to change when small data: n_train and n_test in utils.py, n_components in fisher_feature_extractor.py
folder_name = 'data/'
# folder_name = 'data_small/'

nclasses = 10
classifier = 'svm_ovo'
do_validation = True
validation = 0.2
do_prediction = False

svm_kernel = LinearKernel()
#svm_kernel = LaplacianRBFKernel(1.6)
C = 1

Xtrain, Ytrain, Xtest = load_features(feature_extractor, overwrite_features, overwrite_kpca,
                                        kernel_pca, kernel_pca_kernel, cut_percentage, folder_name)
#Xtrain = numpy.reshape(Xtrain, (Xtrain.shape[0], -1))
#Xtest = numpy.reshape(Xtest, (Xtest.shape[0], -1))
print(Xtrain.shape)
print(Xtest.shape)
assert Xtrain.ndim == 2 and Xtrain.shape[1] == Xtest.shape[1]

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
    K = svm_kernel.build_K(Xtrain)

    if do_validation:
        model = KernelSVMOneVsOneClassifier(nclasses, svm_kernel)
        model.fit(Xtrain, Ytrain, C, validation, K=K, check=True)

    if do_prediction:
        model = KernelSVMOneVsOneClassifier(nclasses, svm_kernel)
        model.fit(Xtrain, Ytrain, C, K=K)
elif classifier == 'svm_ova':
    K = svm_kernel.build_K(Xtrain)

    if do_validation:
        model = KernelSVMOneVsAllClassifier(nclasses, svm_kernel)
        model.fit(Xtrain, Ytrain, C, validation, K=K, check=True)

    if do_prediction:
        model = KernelSVMOneVsAllClassifier(nclasses, svm_kernel)
        model.fit(Xtrain, Ytrain, C, K=K)
else:
    raise Exception("Unknown classifier")

if do_prediction:
    print("Predicting on test data")
    Ytest = model.predict(Xtest)
    write_output(Ytest, 'results/Yte_' + output_suffix + '.csv')
