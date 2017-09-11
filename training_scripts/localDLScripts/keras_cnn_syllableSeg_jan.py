
# from keras.utils.np_utils import to_categorical
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from training_scripts.data_preparation import load_data
from training_scripts.models import jan, model_train

# space = {
#             'filter_density': hp.choice('filter_density', [1, 2, 3, 4]),
#
#             'dropout': hp.uniform('dropout', 0.25, 0.5),
#         }


nlen = 21
input_dim = (80, nlen)

def train_model(filter_density, dropout, input_shape, file_path_model, filename_log):
    """
    train final model save to model path
    """

    filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all.h5'
    filename_labels_train_validation_set = '../../trainingData/labels_train_set_all_syllableSeg_mfccBands2D_old+new.pickle.gz'
    filename_sample_weights = '../../trainingData/sample_weights_syllableSeg_mfccBands2D_old+new.pickle.gz'

    filenames_train, Y_train, sample_weights_train, \
    filenames_validation, Y_validation, sample_weights_validation, \
    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data(filename_labels_train_validation_set,
              filename_sample_weights)

    model_0 = jan(filter_density=filter_density, dropout=dropout, input_shape=input_shape)

    batch_size = 128
    patience = 10

    print(model_0.count_params())

    # hist = model_merged_0.fit(X_train,
    #                         Y_train,
    #                         validation_data=[X_validation, Y_validation],
    #                         class_weight=class_weights,
    #                         callbacks=callbacks,
    #                         sample_weight=sample_weights_X_train,
    #                         epochs=500,
    #                         batch_size=128)
    # nb_epoch = len(hist.history['acc'])
    #
    # model_merged_1 = f_nn_model(filter_density, dropout)
    #
    # model_merged_1.fit(X_train_validation,
    #                     Y_train_validation,
    #                   class_weight=class_weights,
    #                   sample_weight=sample_weights,
    #                     epochs=nb_epoch,
    #                     batch_size=128)
    #
    # model_merged_1.save(file_path_model)

    model_train(model_0, batch_size, patience, input_shape,
                filename_train_validation_set,
                filenames_train, Y_train, sample_weights_train,
                filenames_validation, Y_validation, sample_weights_validation,
                filenames_features, Y_train_validation, sample_weights, class_weights,
                file_path_model, filename_log)

if __name__ == '__main__':

    # uncomment for parameters search
    # trials = Trials()
    # best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
    # print 'best: '
    # print best

    # train the final model
    file_path_model = '../../cnnModels/keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_old+new.h5'
    file_path_log = '../../cnnModels/log/keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_old+new.csv'
    train_model(filter_density=1, dropout=0.2503, input_shape=input_dim, file_path_model=file_path_model, filename_log=file_path_log)
