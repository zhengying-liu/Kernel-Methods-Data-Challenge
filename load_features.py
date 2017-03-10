import numpy
import os

from bag_of_words import BagOfWords
from fisher_feature_extractor import FisherFeatureExtractor
from hog_feature_extractor import HOGFeatureExtractor
from kernel_descriptors_extractor import KernelDescriptorsExtractor
from kernel_pca import KernelPCA
from utils import load_data
from sift_feature_extractor import SIFTFeatureExtractor

def get_feature_extractor(feature_extractor):
    if feature_extractor == 'hog':
        return HOGFeatureExtractor()
    elif feature_extractor == 'hog_fisher':
        return FisherFeatureExtractor(local_feature_extractor_name='hog')
    elif feature_extractor == 'sift':
        return SIFTFeatureExtractor()
    elif feature_extractor == 'sift_fisher':
        return FisherFeatureExtractor(local_feature_extractor_name='sift')
    elif feature_extractor == 'kernel_descriptors':
        return KernelDescriptorsExtractor()
    elif feature_extractor == 'bag_of_words_hog':
        return BagOfWords(local_feature_extractor_name='hog')
    elif feature_extractor == 'raw':
        return None
    else:
        raise Exception("Unknown feature extractor")

def load_features(feature_extractor_name, overwrite_features=True, overwrite_kpca=True,
                    do_kpca=False, kpca_kernel=None, cut_percentage=90, folder_name='data/'):
    Xtrain, Ytrain, Xtest = load_data(folder_name)

    if not overwrite_features and not overwrite_kpca and do_kpca:
        assert kpca_kernel is not None
        kernel_name = kpca_kernel.name
        file_suffix = '_' + feature_extractor_name + '_' + kernel_name + '.npy'

        if os.path.isfile(folder_name + 'Xtrain' + file_suffix) \
                and os.path.isfile(folder_name + 'Xtest' + file_suffix):
            Xtrain = numpy.load(folder_name + 'Xtrain' + file_suffix)
            Xtest = numpy.load(folder_name + 'Xtest' + file_suffix)
            return Xtrain, Ytrain, Xtest

    feature_extractor = get_feature_extractor(feature_extractor_name)
    if feature_extractor_name == 'hog_fisher' or feature_extractor_name == 'sift_fisher':
        if not overwrite_features and os.path.isfile(folder_name + 'Xtrain_' + feature_extractor_name + '.npy') \
                and os.path.isfile(folder_name + 'Xtest_' + feature_extractor_name + '.npy'):
            Xtrain = numpy.load(folder_name + 'Xtrain_' + feature_extractor_name + '.npy')
            Xtest = numpy.load(folder_name + 'Xtest_' + feature_extractor_name + '.npy')
        else:
            Xtrain, V_truncate, gmm = feature_extractor.train(Xtrain)
            Xtest = feature_extractor.predict(Xtest, V_truncate, gmm)
            numpy.save(folder_name + 'Xtrain_' + feature_extractor_name, Xtrain)
            numpy.save(folder_name + 'Xtest_' + feature_extractor_name, Xtest)
    elif feature_extractor_name == 'bag_of_words_hog':
        if not overwrite_features and os.path.isfile(folder_name + 'Xtrain_' + feature_extractor_name + '.npy') \
                and os.path.isfile(folder_name + 'Xtest_' + feature_extractor_name + '.npy'):
            Xtrain = numpy.load(folder_name + 'Xtrain_' + feature_extractor_name + '.npy')
            Xtest = numpy.load(folder_name + 'Xtest_' + feature_extractor_name + '.npy')
        else:
            Xtrain = feature_extractor.extract(Xtrain)
            Xtest = feature_extractor.extract(Xtest)
            feature_extractor.fit(Xtrain)
            Xtrain = feature_extractor.predict(Xtrain)
            Xtest = feature_extractor.predict(Xtest)
            numpy.save(folder_name + 'Xtrain_' + feature_extractor_name, Xtrain)
            numpy.save(folder_name + 'Xtest_' + feature_extractor_name, Xtest)
    elif feature_extractor is not None:
        if not overwrite_features and os.path.isfile(folder_name + 'Xtrain_' + feature_extractor_name + '.npy'):
            Xtrain = numpy.load(folder_name + 'Xtrain_' + feature_extractor_name + '.npy')
        else:
            Xtrain = feature_extractor.predict(Xtrain)
            numpy.save(folder_name + 'Xtrain_' + feature_extractor_name, Xtrain)

        if not overwrite_features and os.path.isfile(folder_name + 'Xtest_' + feature_extractor_name + '.npy'):
            Xtest = numpy.load(folder_name + 'Xtest_' + feature_extractor_name + '.npy')
        else:
            Xtest = feature_extractor.predict(Xtest)
            numpy.save(folder_name + 'Xtest_' + feature_extractor_name, Xtest)

    if do_kpca:
        kpca = KernelPCA(kpca_kernel)
        kpca.fit(Xtrain, cut_percentage=cut_percentage)
        Xtrain = kpca.predict(Xtrain)
        Xtest = kpca.predict(Xtest)

        kernel_name = kpca_kernel.name
        file_suffix = '_' + feature_extractor_name + '_' + kernel_name + '.npy'
        numpy.save(folder_name + 'Xtrain' + file_suffix, Xtrain)
        numpy.save(folder_name + 'Xtest' + file_suffix, Xtest)

    return Xtrain, Ytrain, Xtest
