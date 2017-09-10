from training_scripts.data_preparation import load_data
from training_scripts.models import jordi_model, model_train

nlen = 21
input_dim = (80, nlen)


# space = {
#             'filter_density': hp.choice('filter_density', [1]),
#
#             'dropout': hp.uniform('dropout', 0.25, 0.5),
#
#             'layer2_nodes': hp.choice('layer2_nodes', [20, 32]),
#
#             'pool_n_row': hp.choice('pool_n_row', [3]),
#
#             'pool_n_col': hp.choice('pool_n_col', [1])
#         }



def train_model(filter_density1, filter_density2,
                pool_n_row, pool_n_col,
                dropout, input_shape,
                file_path_model, filename_log):
    """
    train final model save to model path
    """

    folder_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/features_train_set_all_syllableSeg_mfccBands2D_old+new'
    filename_labels_train_validation_set = '../trainingData/labels_train_set_all_syllableSeg_mfccBands2D_old+new.pickle.gz'
    filename_sample_weights = '../trainingData/sample_weights_syllableSeg_mfccBands2D_old+new.pickle.gz'

    filenames_train, Y_train, sample_weights_train, \
    filenames_validation, Y_validation, sample_weights_validation, \
    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data(folder_train_validation_set,
                  filename_labels_train_validation_set,
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

    model_0 = jordi_model(filter_density1, filter_density2,
                            pool_n_row, pool_n_col,
                            dropout, input_shape,
                          'temporal')

    batch_size = 128
    patience = 10

    print(model_0.count_params())

    model_train(model_0, batch_size, patience, input_shape,
                filenames_train, Y_train, sample_weights_train,
                filenames_validation, Y_validation, sample_weights_validation,
                filenames_features, Y_train_validation, sample_weights, class_weights,
                file_path_model, filename_log)
    

if __name__ == '__main__':

    # Uncomment this for parameter search
    # trials = Trials()
    # best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
    # print 'best: '
    # print best

    # train the model
    file_path_model = '../cnnModels/keras.cnn_syllableSeg_jordi_temporal_class_weight_with_conv_dense_mfccBands_2D_old+new.h5'
    file_path_log = '../cnnModels/log/keras.cnn_syllableSeg_jordi_temporal_class_weight_with_conv_dense_mfccBands_2D_old+new.csv'
    train_model(filter_density1=1, filter_density2=1,
                pool_n_row=3, pool_n_col=5, dropout=0.30, input_shape=input_dim,
                file_path_model=file_path_model, filename_log=file_path_log)
