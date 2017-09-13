
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from training_scripts.data_preparation import load_data
from training_scripts.models import jordi_model, model_train

nlen = 21
input_dim = (80, nlen)

# space = {
#             'filter_density': hp.choice('filter_density', [1]),
#
#             'dropout': hp.choice('dropout', [0.3]),
#
#             'layer2_nodes': hp.choice('layer2_nodes', [20, 32]),
#
#             'pool_n_row': hp.choice('pool_n_row', [1,3,5,7]),
#
#             'pool_n_col': hp.choice('pool_n_col', [1,3,5,7])
#         }


# def f_nn(params):
#     print ('Params testing: ', params)
#
#     model_merged = f_nn_model(params['filter_density'], params['layer2_nodes'], params['pool_n_row'], params['pool_n_col'], params['dropout'])
#
#     callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]
#
#     hist = model_merged.fit([X_train, X_train, X_train, X_train, X_train, X_train],
#                             Y_train,
#                             validation_data=[[X_validation,X_validation,X_validation,X_validation,X_validation,X_validation], Y_validation],
#                             # validation_split=0.2,
#                             class_weight=class_weights,
#                             sample_weight=sample_weights_X_train,
#                               callbacks=callbacks,
#                               nb_epoch=500,
#                               batch_size=124,
#                               verbose=0)
#
#     # score, acc = model_merged.evaluate([X_validation, X_validation, X_validation, X_validation, X_validation, X_validation], Y_validation, batch_size = 128, verbose = 0)
#     acc = hist.history['val_acc'][-1]
#     print('Test accuracy:', acc, 'nb_epoch:', len(hist.history['acc']))
#     return {'loss': -acc, 'status': STATUS_OK}


def train_model(filter_density_1, filter_density_2,
                pool_n_row, pool_n_col,
                dropout, input_shape,
                file_path_model, filename_log):
    """
    train final model save to model path
    """

    filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all.h5'
    filename_labels_train_validation_set = '../trainingData/labels_train_set_all_syllableSeg_mfccBands2D_old+new.pickle.gz'
    filename_sample_weights = '../trainingData/sample_weights_syllableSeg_mfccBands2D_old+new.pickle.gz'

    filenames_train, Y_train, sample_weights_train, \
    filenames_validation, Y_validation, sample_weights_validation, \
    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data(filename_labels_train_validation_set,
                  filename_sample_weights)

    # model_merged_0 = f_nn_model(filter_density, layer2_nodes, pool_n_row, pool_n_col, dropout)
    #
    # callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]
    #
    # hist = model_merged_0.fit([X_train, X_train, X_train, X_train, X_train, X_train],
    #                         Y_train,
    #                         validation_data=[
    #                             [X_validation, X_validation, X_validation, X_validation, X_validation, X_validation],
    #                             Y_validation],
    #                         # validation_split=0.2,
    #                         class_weight=class_weights,
    #                         sample_weight=sample_weights_X_train,
    #                         callbacks=callbacks,
    #                         nb_epoch=500,
    #                         batch_size=128,
    #                         verbose=0)
    #
    # nb_epoch = len(hist.history['val_acc'])
    #
    # model_merged_1 = f_nn_model(filter_density, layer2_nodes, pool_n_row, pool_n_col, dropout)
    #
    # print(model_merged_1.count_params())
    #
    # hist = model_merged_1.fit([X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation],
    #                             Y_train_validation,
    #                           class_weight=class_weights,
    #                           sample_weight=sample_weights,
    #                             nb_epoch=nb_epoch,
    #                             batch_size=128)
    #
    # model_merged_1.save(file_path_model)

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

    # uncomment below for parameters search
    # trials = Trials()
    # best = fmin(f_nn, space, algo=tpe.suggest, max_evals=16, trials=trials)
    # print 'best: '
    # print best

    # train the final model

    file_path_model = '../cnnModels/keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_old+new.h5'
    file_path_model = '../cnnModels/log/keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_old+new.csv'

    train_model(filter_density_1=1, filter_density_2=1,
                pool_n_row=5, pool_n_col=3,
                dropout=0.3, input_shape=input_dim,
                file_path_model=file_path_model, filename_log=file_path_model)
