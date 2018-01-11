import cPickle
import gzip
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import os

def load_data(filename_labels_train_validation_set,
              filename_sample_weights):

    # load training and validation data

    with gzip.open(filename_labels_train_validation_set, 'rb') as f:
        Y_train_validation = cPickle.load(f)

    with gzip.open(filename_sample_weights, 'rb') as f:
        sample_weights = cPickle.load(f)

    print(len(Y_train_validation[Y_train_validation==0]), len(Y_train_validation[Y_train_validation==1]))

    # print(Y_train_validation)

    # filenames_features = [os.path.join(folder_train_validation_set, f)
    #                       for f in os.listdir(folder_train_validation_set)
    #                       if os.path.isfile(os.path.join(folder_train_validation_set, f)) and f!='.DS_Store']

    # this is the filename indices
    indices_features = range(len(Y_train_validation))
    #
    # # remove features not exist
    # index_2_remove = [77983, 77928] # these two indices are not presented in training set
    # Y_train_validation = np.delete(Y_train_validation, index_2_remove)
    # sample_weights = np.delete(sample_weights, index_2_remove)
    # filenames_features = np.delete(filenames_features, index_2_remove)


    # print(indices_features[:20])

    class_weights = compute_class_weight('balanced',[0,1],Y_train_validation)

    # print(class_weights)

    class_weights = {0:class_weights[0], 1:class_weights[1]}

    # X_train_validation                              = np.array([[X_train_validation[ii], sample_weights[ii]] for ii in range(len(X_train_validation))])
    filenames_features_sample_weights               = np.array([[indices_features[ii], sample_weights[ii]] for ii in range(len(indices_features))])

    filenames_sample_weights_train, filenames_sample_weights_validation, Y_train, Y_validation    = \
        train_test_split(filenames_features_sample_weights, Y_train_validation, test_size=0.1, stratify=Y_train_validation)

    indices_train                               = [int(xt[0]) for xt in filenames_sample_weights_train]
    sample_weights_train                          = np.array([xt[1] for xt in filenames_sample_weights_train])
    indices_validation                          = [int(xv[0]) for xv in filenames_sample_weights_validation]
    sample_weights_validation                     = np.array([xv[1] for xv in filenames_sample_weights_validation])

    # print(indices_train[:10])


    # Y_train_validation                              = to_categorical(Y_train_validation)
    # Y_train                                         = to_categorical(Y_train)
    # Y_validation                                    = to_categorical(Y_validation)

    # print(sample_weights_train.shape, len(indices_train), Y_train.shape)
    # print(sample_weights_validation.shape, len(indices_validation), Y_validation.shape)

    return indices_train, Y_train, sample_weights_train, \
           indices_validation, Y_validation, sample_weights_validation, \
           indices_features, Y_train_validation, sample_weights, class_weights


def load_data_schluter(filename_labels_train_validation_set,
                        filename_sample_weights):

    # load training and validation data

    with gzip.open(filename_labels_train_validation_set, 'rb') as f:
        Y_train_validation = cPickle.load(f)

    with gzip.open(filename_sample_weights, 'rb') as f:
        sample_weights = cPickle.load(f)

    print(len(Y_train_validation[Y_train_validation==0]), len(Y_train_validation[Y_train_validation==1]))

    # this is the filename indices
    indices_train_validation = range(len(Y_train_validation))

    print(indices_train_validation[:20])

    class_weights = compute_class_weight('balanced',[0,1],Y_train_validation)

    print(class_weights)

    class_weights = {0:class_weights[0], 1:class_weights[1]}


    # Y_train_validation                              = to_categorical(Y_train_validation)
    # Y_train                                         = to_categorical(Y_train)
    # Y_validation                                    = to_categorical(Y_validation)


    return indices_train_validation, Y_train_validation, sample_weights, class_weights