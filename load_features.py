import numpy
import os

from fisher_feature_extractor import FisherFeatureExtractor
from hog_feature_extractor import HOGFeatureExtractor
from kernel_descriptors_extractor import KernelDescriptorsExtractor
from kernel_pca import KernelPCA
from utils import load_data

def get_feature_extractor(feature_extractor):
    if feature_extractor == 'hog':
        return HOGFeatureExtractor()
    elif feature_extractor == 'hog_fisher':
        return FisherFeatureExtractor(nclasses=5)
    elif feature_extractor == 'kernel_descriptors':
        return KernelDescriptorsExtractor()
    elif feature_extractor == 'raw':
        return None
    else:
        raise Exception("Unknown feature extractor")

def load_features(feature_extractor_name, overwrite, kernel_pca=False, kernel_pca_kernel=None,
                    cut_percentage=90):
    Xtrain, Ytrain, Xtest = load_data()

    if not overwrite and kernel_pca:
        assert kernel_pca_kernel is not None
        kernel_name = kernel_pca_kernel.name
        file_suffix = '_' + feature_extractor_name + '_' + kernel_name + '.npy'

        if os.path.isfile('data/Xtrain' + file_suffix) \
                and os.path.isfile('data/Xtest' + file_suffix):
            Xtrain = numpy.load('data/Xtrain' + file_suffix)
            Xtest = numpy.load('data/Xtest' + file_suffix)
            return Xtrain, Ytrain, Xtest

    feature_extractor = get_feature_extractor(feature_extractor_name)
    if feature_extractor is not None:
        if not overwrite and os.path.isfile('data/Xtrain_' + feature_extractor_name + '.npy'):
            Xtrain = numpy.load('data/Xtrain_' + feature_extractor_name + '.npy')
        else:
            Xtrain = feature_extractor.predict(Xtrain)
            numpy.save('data/Xtrain_' + feature_extractor_name, Xtrain)

        if not overwrite and os.path.isfile('data/Xtest_' + feature_extractor_name + '.npy'):
            Xtest = numpy.load('data/Xtest_' + feature_extractor_name + '.npy')
        else:
            Xtest = feature_extractor.predict(Xtest)
            numpy.save('data/Xtest_' + feature_extractor_name, Xtest)

    if kernel_pca:
        kpca = KernelPCA(kernel_pca_kernel)
        X = numpy.concatenate((Xtrain, Xtest), axis=0)
        kpca.fit(X, cut_percentage=cut_percentage)

        X = kpca.predict()
        ntrain = Xtrain.shape[0]
        Xtrain = X[:ntrain, :]
        Xtest = X[ntrain:, :]

        kernel_name = kernel_pca_kernel.name
        file_suffix = '_' + feature_extractor_name + '_' + kernel_name + '.npy'
        numpy.save('data/Xtrain' + file_suffix, Xtrain)
        numpy.save('data/Xtest' + file_suffix, Xtest)

    return Xtrain, Ytrain, Xtest
