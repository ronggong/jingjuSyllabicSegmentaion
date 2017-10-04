
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation import load_data
from models import jordi_model, model_train

nlen = 21
input_dim = (80, nlen)


def train_model(filter_density_1, filter_density_2,
                pool_n_row, pool_n_col,
                dropout, input_shape,
                file_path_model, filename_log):
    """
    train final model save to model path
    """

    filename_train_validation_set = '/scratch/rgongcnnSyllableSeg_timbral/syllableSeg/feature_all_artist_filter_split.h5'
    filename_labels_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_train_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_split.pickle.gz'
    filename_sample_weights = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_syllableSeg_mfccBands2D_old+new_artist_filter_split.pickle.gz'

    filenames_train, Y_train, sample_weights_train, \
    filenames_validation, Y_validation, sample_weights_validation, \
    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data(filename_labels_train_validation_set,
              filename_sample_weights)


    model_0 = jordi_model(filter_density_1, filter_density_2,
                            pool_n_row, pool_n_col,
                            dropout, input_shape,
                          'timbral')

    batch_size = 128
    patience = 10

    print(model_0.count_params())

    model_train(model_0, batch_size, patience, input_shape,
                filename_train_validation_set,
                filenames_train, Y_train, sample_weights_train,
                filenames_validation, Y_validation, sample_weights_validation,
                filenames_features, Y_train_validation, sample_weights, class_weights,
                file_path_model, filename_log)


if __name__ == '__main__':


    file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_artist_filter_split.h5'
    file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_artist_filter_split.csv'

    train_model(filter_density_1=1, filter_density_2=1,
                pool_n_row=5, pool_n_col=3,
                dropout=0.3, input_shape=input_dim,
                file_path_model=file_path_model, filename_log=file_path_log)
