from src.schluterParser import annotationCvParser
from src.filePathSchulter import *
from src.parameters import *
from trainingSampleCollectionSchluter import getRecordings
from trainingSampleCollection import featureReshape
import gzip, cPickle
from os.path import isfile
import numpy as np
import h5py

def getTrainingFilenames(annotation_path, cv_filename):
    """
    annotation filenames - cv test filenames
    :param annotation_path:
    :param cv_filename:
    :return:
    """
    annotation_fns = getRecordings(annotation_path)
    test_fns = annotationCvParser(cv_filename)
    train_fns = [x for x in annotation_fns if x not in test_fns]
    return train_fns

def concatenateFeatureLabelSampleweights(train_fns, schluter_feature_data_path):
    """
    concatenate feature label and sample weights
    :param train_fns:
    :return:
    """
    label_all = []
    sample_weights_all = []
    for fn in train_fns:
        sample_weights_fn = join(schluter_feature_data_path, 'sample_weights_' + fn + '.pickle.gz')
        label_fn = join(schluter_feature_data_path, 'label_'+fn+'.pickle.gz')

        if not isfile(sample_weights_fn):
            print(sample_weights_fn, 'not found.')
            continue
        if not isfile(label_fn):
            print(label_fn, 'not found.')
            continue


        with gzip.open(sample_weights_fn, 'rb') as f:
            sample_weights = cPickle.load(f)
            sample_weights_all.append(sample_weights)

        with gzip.open(label_fn, 'rb') as f:
            label = cPickle.load(f)
            label_all.append(label)

    sample_weights_all = np.concatenate(sample_weights_all)
    label_all = np.concatenate(label_all)


    nDims = 80*21
    feature_all = np.zeros((len(label_all), nDims), dtype='float32')

    idx_start = 0
    for fn in train_fns:
        # print('Concatenating feature ...', fn, 'idx start', idx_start)
        feature_fn = join(schluter_feature_data_path, 'feature_'+fn+'.h5')
        if not isfile(feature_fn):
            print(feature_fn, 'not found.')
            continue
        feature = h5py.File(feature_fn, 'r')
        dim_feature = feature['feature_all'].shape[0]
        feature_all[idx_start:(idx_start+dim_feature), :] = feature['feature_all']
        idx_start += dim_feature
        feature.flush()
        feature.close()

    feature_all = featureReshape(feature_all, nlen=varin['nlen'])

    return feature_all, label_all, sample_weights_all

def saveFeatureLabelSampleweights(feature_all, label_all, sample_weights,
                                  feature_fn, label_fn, sample_weights_fn):
    h5f = h5py.File(feature_fn, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    cPickle.dump(label_all,
                 gzip.open(
                     label_fn,
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights,
                 gzip.open(
                     sample_weights_fn,
                     'wb'), cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    test_cv_filename = join(schluter_cv_path, '8-fold_cv_random_0.fold')
    train_fns = getTrainingFilenames(schluter_annotations_path, test_cv_filename)
    print(len(train_fns))
    feature_all, label_all, sample_weights_all = concatenateFeatureLabelSampleweights(train_fns)
    print(feature_all.shape)
    print(len(label_all))
    print(len(sample_weights_all))
