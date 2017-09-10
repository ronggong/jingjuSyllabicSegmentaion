import numpy as np
import os
from keras import backend as K

K.set_image_dim_ordering('th')

from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import cPickle, pickle
import gzip
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

filename_out = '/homedtic/rgong/cnnSyllableSeg/out/syllableSeg_horizontal_vertical_filter_node07.txt'
if os.path.isfile(filename_out):
    file = open(filename_out,'a')
else:
    file = open(filename_out,'w')
file.write("debug")
file.close()

# load training and validation data
filename_train_validation_set = '/scratch/rgongcnnSyllableSeg_node07/syllableSeg/train_set_all_syllableSeg_mfccBands2D.pickle.gz'
filename_sample_weights = '/scratch/rgongcnnSyllableSeg_node07/syllableSeg/sample_weights_syllableSeg_mfccBands2D.pickle.gz'

with gzip.open(filename_train_validation_set, 'rb') as f:
    X_train_validation, Y_train_validation = cPickle.load(f)

with gzip.open(filename_sample_weights, 'rb') as f:
    sample_weights = cPickle.load(f)

print(X_train_validation.shape, Y_train_validation.shape, sample_weights.shape)
print(len(Y_train_validation[Y_train_validation==0]), len(Y_train_validation[Y_train_validation==1]))
class_weights = compute_class_weight('balanced',[0,1],Y_train_validation)
print(class_weights)
class_weights = {0:class_weights[0], 1:class_weights[1]}

X_train_validation                              = np.array([[X_train_validation[ii], sample_weights[ii]] for ii in range(len(X_train_validation))])

X_train, X_validation, Y_train, Y_validation    = train_test_split(X_train_validation, Y_train_validation, test_size=0.2, stratify=Y_train_validation)
sample_weights_X_train                          = np.array([xt[1] for xt in X_train])
X_train_validation                              = np.array([xtv[0] for xtv in X_train_validation], dtype='float32')
X_train                                         = np.array([xt[0] for xt in X_train], dtype='float32')
X_validation                                    = np.array([xv[0] for xv in X_validation], dtype='float32')

Y_train_validation                              = to_categorical(Y_train_validation)
Y_train                                         = to_categorical(Y_train)
Y_validation                                    = to_categorical(Y_validation)

space = {
            'filter_density': hp.choice('filter_density', [1]),

            'dropout': hp.uniform('dropout', 0.25, 0.5),

            'pool_n_row': hp.choice('pool_n_row', [1,3,5,7]),

            'pool_n_col': hp.choice('pool_n_col', [1,3,5,7])
        }


from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten, Merge, ELU
from keras.regularizers import l2


def createModel(model, reshape_dim, input_dim, num_filter, height_filter, width_filter, filter_density, pool_n_row, pool_n_col, dropout):
    model.add(Reshape(reshape_dim, input_shape=input_dim))
    model.add(
        Convolution2D(num_filter * filter_density, height_filter, width_filter, border_mode='same', input_shape=reshape_dim, dim_ordering='th',
                      init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())

    if pool_n_row == 'all' and pool_n_col == 'all':
        model.add(MaxPooling2D(pool_size=(model.output_shape[2], model.output_shape[3]), border_mode='same',
                                 dim_ordering='th'))
    elif pool_n_row == 'all' and pool_n_col != 'all':
        model.add(MaxPooling2D(pool_size=(model.output_shape[2], pool_n_col), border_mode='same',
                                 dim_ordering='th'))
    elif pool_n_row != 'all' and pool_n_col == 'all':
        model.add(MaxPooling2D(pool_size=(pool_n_row, model.output_shape[3]), border_mode='same',
                                 dim_ordering='th'))
    else:
        model.add(MaxPooling2D(pool_size=(pool_n_row, pool_n_col), border_mode='same', dim_ordering='th'))
    model.add(Dropout(dropout))
#    model.add(Flatten())
    return model


def f_nn_model(filter_density, pool_n_row, pool_n_col, dropout):

    nlen = 21
    reshape_dim = (1, 80, nlen)
    input_dim = (80, nlen)
    channel_axis = 1


    model_1 = Sequential()
    model_1 = createModel(model_1, reshape_dim, input_dim, 12, 50, 1, filter_density, pool_n_row, pool_n_col, dropout)

    model_2 = Sequential()
    model_2 = createModel(model_2, reshape_dim, input_dim, 6, 50, 5, filter_density, pool_n_row, pool_n_col, dropout)

    model_3 = Sequential()
    model_3 = createModel(model_3, reshape_dim, input_dim, 3, 50, 10, filter_density, pool_n_row, pool_n_col, dropout)


    model_4 = Sequential()
    model_4 = createModel(model_4, reshape_dim, input_dim, 12, 70, 1, filter_density, pool_n_row, pool_n_col, dropout)

    model_5 = Sequential()
    model_5 = createModel(model_5, reshape_dim, input_dim, 6, 70, 5, filter_density, pool_n_row, pool_n_col, dropout)

    model_6 = Sequential()
    model_6 = createModel(model_6, reshape_dim, input_dim, 3, 70, 10, filter_density, pool_n_row, pool_n_col, dropout)

    model_7 = Sequential()
    model_7 = createModel(model_7, reshape_dim, input_dim, 12, 1, 7, filter_density, pool_n_row, pool_n_col, dropout)

    model_8 = Sequential()
    model_8 = createModel(model_8, reshape_dim, input_dim, 6, 3, 7, filter_density, pool_n_row, pool_n_col, dropout)

    model_9 = Sequential()
    model_9 = createModel(model_9, reshape_dim, input_dim, 3, 5, 7, filter_density, pool_n_row, pool_n_col, dropout)

    model_10 = Sequential()
    model_10 = createModel(model_10, reshape_dim, input_dim, 12, 1, 12, filter_density, pool_n_row, pool_n_col, dropout)

    model_11 = Sequential()
    model_11 = createModel(model_11, reshape_dim, input_dim, 6, 3, 12, filter_density, pool_n_row, pool_n_col, dropout)

    model_12 = Sequential()
    model_12 = createModel(model_12, reshape_dim, input_dim, 3, 5, 12, filter_density, pool_n_row, pool_n_col, dropout)

    merged = Merge([model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10, model_11, model_12], mode='concat', concat_axis=1)

    model_merged = Sequential()
    model_merged.add(merged)
    
    # conv 2 layers
    model_merged.add(Convolution2D(20 * filter_density, 3, 3, border_mode='valid', dim_ordering='th',
                                   init='he_uniform', W_regularizer=l2(1e-5)))
    # model_merged.add(BatchNormalization(axis=channel_axis, mode=0))

    model_merged.add(ELU())
    model_merged.add(MaxPooling2D(pool_size=(3, 1), border_mode='valid', dim_ordering='th'))
    model_merged.add(Dropout(dropout))
    model_merged.add(Flatten())


    # dense
    model_merged.add(Dense(output_dim=256, init='he_uniform', W_regularizer=l2(1e-5)))
    model_merged.add(ELU())
    model_merged.add(Dropout(dropout))

    model_merged.add(Dense(output_dim=2))
    model_merged.add(Activation("softmax"))

    # optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
    optimizer = Adam()

    model_merged.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])

    return model_merged


def f_nn(params):
    print ('Params testing: ', params)

    model_merged = f_nn_model(params['filter_density'], params['pool_n_row'], params['pool_n_col'], params['dropout'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]

    hist = model_merged.fit([X_train, X_train, X_train, X_train, X_train, X_train, X_train, X_train, X_train, X_train, X_train, X_train],
                            Y_train,
                            validation_data=[[X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation], Y_validation],
                            # validation_split=0.2,
                            class_weight=class_weights,
                            sample_weight=sample_weights_X_train,
                              callbacks=callbacks,
                              nb_epoch=500,
                              batch_size=64,
                              verbose=0)

    # score, acc = model_merged.evaluate([X_validation, X_validation, X_validation, X_validation, X_validation, X_validation], Y_validation, batch_size = 128, verbose = 0)
    acc = hist.history['val_acc'][-1]
    nb_epoch = len(hist.history['acc'])

    filename_out = '/homedtic/rgong/cnnSyllableSeg/out/syllableSeg_horizontal_vertical_filter_node07.txt'
    if os.path.isfile(filename_out):
        file = open(filename_out,'a')
    else:
        file = open(filename_out,'w')

    file.write(str(params)+str(acc)+'_'+str(nb_epoch)+'\n')
    file.close()

    print('Test accuracy:', acc, 'nb_epoch:', len(hist.history['acc']))
    return {'loss': -acc, 'status': STATUS_OK}


def train_model(filter_density, pool_n_row, pool_n_col, dropout, nb_epoch, file_path_model):
    """
    train final model save to model path
    """
    
    model_merged_0 = f_nn_model(filter_density, pool_n_row, pool_n_col, dropout)

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]

    hist = model_merged_0.fit([X_train, X_train, X_train, X_train, X_train, X_train,X_train, X_train, X_train, X_train, X_train, X_train],
                           Y_train,
                           validation_data=[[X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation,X_validation], Y_validation],
                           # validation_split=0.2,
                           class_weight=class_weights,
                           sample_weight=sample_weights_X_train,
                           callbacks=callbacks,
                           nb_epoch=500,
                           batch_size=128,
                           verbose=0)

    nb_epoch = len(hist.history['val_acc'])

    model_merged_1 = f_nn_model(filter_density, pool_n_row, pool_n_col, dropout)

    print(model_merged_1.count_params())

    hist = model_merged_1.fit([X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation, X_train_validation],
                                Y_train_validation,
                              class_weight=class_weights,
                              sample_weight=sample_weights,
                                nb_epoch=nb_epoch,
                                batch_size=128)

    model_merged_1.save(file_path_model)


if __name__ == '__main__':

    # parameters search
    # trials = Trials()
    # best = fmin(f_nn, space, algo=tpe.suggest, max_evals=40, trials=trials)
    # print 'best: '
    # print best

    # train the final model

    file_path_model = '/scratch/rgongcnnSyllableSeg_node07/out/keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_horizontal_timbral_filter_layer2_20_mfccBands_2D_all_optim.h5'
    train_model(filter_density=1, pool_n_row=5, pool_n_col=5, dropout=0.30, nb_epoch=101, file_path_model=file_path_model)