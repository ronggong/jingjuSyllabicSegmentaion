import cPickle
import gzip
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import os

def load_data(folder_train_validation_set,
              filename_labels_train_validation_set,
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

    filenames_features = np.array([os.path.join(folder_train_validation_set, str(ii)+'.pickle.gz')
                          for ii in range(len(sample_weights))])

    # remove features not exist
    index_2_remove = [77983, 77928] # these two indices are not presented in training set
    Y_train_validation = np.delete(Y_train_validation, index_2_remove)
    sample_weights = np.delete(sample_weights, index_2_remove)
    filenames_features = np.delete(filenames_features, index_2_remove)

    print(filenames_features[:20])

    class_weights = compute_class_weight('balanced',[0,1],Y_train_validation)

    print(class_weights)

    class_weights = {0:class_weights[0], 1:class_weights[1]}

    # X_train_validation                              = np.array([[X_train_validation[ii], sample_weights[ii]] for ii in range(len(X_train_validation))])
    filenames_features_sample_weights               = np.array([[filenames_features[ii], sample_weights[ii]] for ii in range(len(filenames_features))])

    filenames_sample_weights_train, filenames_sample_weights_validation, Y_train, Y_validation    = \
        train_test_split(filenames_features_sample_weights, Y_train_validation, test_size=0.2, stratify=Y_train_validation)

    filenames_train                               = [xt[0] for xt in filenames_sample_weights_train]
    sample_weights_train                          = np.array([xt[1] for xt in filenames_sample_weights_train])
    filenames_validation                          = [xv[0] for xv in filenames_sample_weights_validation]
    sample_weights_validation                     = np.array([xv[1] for xv in filenames_sample_weights_validation])

    print(filenames_train[:10])


    # Y_train_validation                              = to_categorical(Y_train_validation)
    # Y_train                                         = to_categorical(Y_train)
    # Y_validation                                    = to_categorical(Y_validation)

    print(sample_weights_train.shape, len(filenames_train), Y_train.shape)
    print(sample_weights_validation.shape, len(filenames_validation), Y_validation.shape)

    return filenames_train, Y_train, sample_weights_train, \
           filenames_validation, Y_validation, sample_weights_validation, \
           filenames_features, Y_train_validation, sample_weights, class_weights