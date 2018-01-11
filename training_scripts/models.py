from os import remove
from os.path import basename

import numpy as np
from keras.callbacks import Callback
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D
from keras.layers import Dropout, Dense, Flatten, ELU, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2

from feature_generator import generator
from data_preparation import load_data_schluter, load_data

def jan(filter_density, dropout, input_shape, batchNorm=False):

    reshape_dim = (1, input_shape[0], input_shape[1])

    model_1 = Sequential()

    if batchNorm:
        model_1.add(BatchNormalization(axis=1, input_shape=reshape_dim))

    model_1.add(Conv2D(int(10 * filter_density), (3, 7), padding="valid",
                    input_shape=reshape_dim, data_format="channels_first",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(3, 1), padding='valid',data_format="channels_first"))

    model_1.add(Conv2D(int(20 * filter_density), (3, 3), padding="valid", data_format="channels_first",
                       kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(3, 1), padding='valid',data_format="channels_first"))

    model_1.add(Dropout(dropout)) # test Schluter dataset, comment in jingju dataset

    model_1.add(Flatten())

    model_1.add(Dense(units=256, kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    
    model_1.add(Dropout(dropout))

    model_1.add(Dense(1, activation='sigmoid'))
    # model_1.add(Activation("softmax"))

    optimizer = Adam()

    model_1.compile(loss='binary_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])

    print(model_1.summary())

    return model_1


def jan_original(filter_density, dropout, input_shape, batchNorm=False, dense_activation='relu', channel=1):
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    model_1 = Sequential()

    if batchNorm:
        model_1.add(BatchNormalization(axis=1, input_shape=reshape_dim))

    model_1.add(Conv2D(int(10 * filter_density), (3, 7), padding="valid",
                       input_shape=reshape_dim,
                       data_format=channel_order, activation='relu'))
    model_1.add(MaxPooling2D(pool_size=(3, 1), padding='valid', data_format=channel_order))

    model_1.add(Conv2D(int(20 * filter_density), (3, 3), padding="valid",
                       data_format=channel_order, activation='relu'))
    model_1.add(MaxPooling2D(pool_size=(3, 1), padding='valid', data_format=channel_order))

    if dropout:
        model_1.add(Dropout(dropout))  # test Schluter dataset, comment in jingju dataset

    model_1.add(Flatten())

    model_1.add(Dense(units=256, activation=dense_activation))
    # model_1.add(ELU())

    if dropout:
        model_1.add(Dropout(dropout))

    model_1.add(Dense(1, activation='sigmoid'))
    # model_1.add(Activation("softmax"))

    # optimizer = SGD(lr=0.05, momentum=0.45, decay=0.0, nesterov=False)
    optimizer = Adam()

    model_1.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

    print(model_1.summary())

    return model_1

def jan_deep(filter_density, dropout, input_shape):

    reshape_dim = (1, input_shape[0], input_shape[1])

    model_1 = Sequential()
    model_1.add(Conv2D(int(16 * filter_density), (3, 7), padding="valid",
                    input_shape=reshape_dim, data_format="channels_first",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    # model_1.add(MaxPooling2D(pool_size=(2, 2), padding='valid',data_format="channels_first"))

    model_1.add(Conv2D(int(16 * filter_density), (3, 3), padding="valid", data_format="channels_first",
                       kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(3, 2), padding='valid',data_format="channels_first"))

    model_1.add(Conv2D(int(32 * filter_density), (3, 3), padding="valid", data_format="channels_first",
                       kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    # model_1.add(MaxPooling2D(pool_size=(3, 1), padding='valid', data_format="channels_first"))

    model_1.add(Conv2D(int(32 * filter_density), (3, 3), padding="valid", data_format="channels_first",
                       kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(3, 2), padding='valid', data_format="channels_first"))

    model_1.add(Flatten())

    model_1.add(Dropout(dropout))

    model_1.add(Dense(units=256, kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(Dropout(dropout))

    model_1.add(Dense(units=32, kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(Dropout(dropout))

    model_1.add(Dense(1, activation='sigmoid'))
    # model_1.add(Activation("softmax"))

    optimizer = Adam()

    model_1.compile(loss='binary_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])

    print(model_1.summary())

    return model_1


def createModel(input, num_filter, height_filter, width_filter, filter_density, pool_n_row,
                pool_n_col, dropout):

    x = Conv2D(int(num_filter * filter_density), (height_filter, width_filter), padding="same",
                       data_format="channels_first",
                       kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5))(input)
    x = ELU()(x)

    output_shape = K.int_shape(x)

    if pool_n_row == 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], output_shape[3]), padding='same', data_format="channels_first")(x)
    elif pool_n_row == 'all' and pool_n_col != 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], pool_n_col), padding='same', data_format="channels_first")(x)
    elif pool_n_row != 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(pool_n_row, output_shape[3]), padding='same', data_format="channels_first")(x)
    else:
        x = MaxPooling2D(pool_size=(pool_n_row, pool_n_col), padding='same', data_format="channels_first")(x)
    x = Dropout(dropout)(x)
    #    model.add(Flatten())
    return x

def createModel_schluter(input, num_filter, height_filter, width_filter, filter_density, pool_n_row,
                pool_n_col, dropout):
    """
    original Schluter relu activation, no dropout
    :param input:
    :param num_filter:
    :param height_filter:
    :param width_filter:
    :param filter_density:
    :param pool_n_row:
    :param pool_n_col:
    :param dropout:
    :return:
    """

    x = Conv2D(int(num_filter * filter_density), (height_filter, width_filter), padding="same",
                       data_format="channels_first",
                       activation='relu')(input)

    output_shape = K.int_shape(x)

    if pool_n_row == 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], output_shape[3]), padding='same', data_format="channels_first")(x)
    elif pool_n_row == 'all' and pool_n_col != 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], pool_n_col), padding='same', data_format="channels_first")(x)
    elif pool_n_row != 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(pool_n_row, output_shape[3]), padding='same', data_format="channels_first")(x)
    else:
        x = MaxPooling2D(pool_size=(pool_n_row, pool_n_col), padding='same', data_format="channels_first")(x)
    #    model.add(Flatten())
    return x

def createModel_schluter_valid(input, num_filter, height_filter, width_filter, filter_density, pool_n_row,
                pool_n_col, dropout):
    """
    original Schluter relu activation, no dropout
    :param input:
    :param num_filter:
    :param height_filter:
    :param width_filter:
    :param filter_density:
    :param pool_n_row:
    :param pool_n_col:
    :param dropout:
    :return:
    """

    x = ZeroPadding2D(padding=(0, int(width_filter/2)),  data_format="channels_first")(input)

    x = Conv2D(int(num_filter * filter_density), (height_filter, width_filter), padding="valid",
                       data_format="channels_first",
                       activation='relu')(x)

    output_shape = K.int_shape(x)

    if pool_n_row == 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], output_shape[3]), padding='same', data_format="channels_first")(x)
    elif pool_n_row == 'all' and pool_n_col != 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], pool_n_col), padding='same', data_format="channels_first")(x)
    elif pool_n_row != 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(pool_n_row, output_shape[3]), padding='same', data_format="channels_first")(x)
    else:
        x = MaxPooling2D(pool_size=(pool_n_row, pool_n_col), padding='same', data_format="channels_first")(x)
    #    model.add(Flatten())
    return x

def timbral_layer(filter_density_layer1, pool_n_row, pool_n_col, dropout, input_dim):
    reshape_dim = (1, input_dim[0], input_dim[1])

    input = Input(shape=reshape_dim)

    x_1 = createModel(input, 12, 50, 1, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_2 = createModel(input, 6, 50, 5, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_3 = createModel(input, 3, 50, 10, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_4 = createModel(input, 12, 70, 1, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_5 = createModel(input, 6, 70, 5, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_6 = createModel(input, 3, 70, 10, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    merged = concatenate([x_1, x_2, x_3, x_4, x_5, x_6], axis=1)

    return input, merged

def temporal_layer(filter_density_layer1, pool_n_row, pool_n_col, dropout, input_dim):
    reshape_dim = (1, input_dim[0], input_dim[1])

    input = Input(shape=reshape_dim)

    x_1 = createModel(input, 12, 1, 7, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_2 = createModel(input, 6, 3, 7, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_3 = createModel(input, 3, 5, 7, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_4 = createModel(input, 12, 1, 12, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_5 = createModel(input, 6, 3, 12, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_6 = createModel(input, 3, 5, 12, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    merged = concatenate([x_1, x_2, x_3, x_4, x_5, x_6], axis=1)

    return input, merged

def timbral_layer_schluter(filter_density_layer1, pool_n_row, pool_n_col, dropout, input_dim):
    reshape_dim = (1, input_dim[0], input_dim[1])

    input = Input(shape=reshape_dim)

    x_1 = createModel_schluter(input, 12, 50, 1, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_2 = createModel_schluter(input, 6, 50, 5, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_3 = createModel_schluter(input, 3, 50, 10, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_4 = createModel_schluter(input, 12, 70, 1, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_5 = createModel_schluter(input, 6, 70, 5, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_6 = createModel_schluter(input, 3, 70, 10, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    merged = concatenate([x_1, x_2, x_3, x_4, x_5, x_6], axis=1)

    return input, merged

def timbral_layer_schluter_valid(filter_density_layer1, pool_n_row, pool_n_col, dropout, input_dim):
    reshape_dim = (1, input_dim[0], input_dim[1])

    input = Input(shape=reshape_dim)

    x_1 = createModel_schluter_valid(input, 36, 50, 1, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_2 = createModel_schluter_valid(input, 36, 50, 5, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_3 = createModel_schluter_valid(input, 36, 50, 9, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_4 = createModel_schluter_valid(input, 36, 70, 1, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_5 = createModel_schluter_valid(input, 36, 70, 5, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_6 = createModel_schluter_valid(input, 36, 70, 9, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    merged = concatenate([x_1, x_2, x_3, x_4, x_5, x_6], axis=2)

    return input, merged

def temporal_layer_schluter(filter_density_layer1, pool_n_row, pool_n_col, dropout, input_dim):
    reshape_dim = (1, input_dim[0], input_dim[1])

    input = Input(shape=reshape_dim)

    x_1 = createModel_schluter(input, 12, 1, 7, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_2 = createModel_schluter(input, 6, 3, 7, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_3 = createModel_schluter(input, 3, 5, 7, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_4 = createModel_schluter(input, 12, 1, 12, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_5 = createModel_schluter(input, 6, 3, 12, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_6 = createModel_schluter(input, 3, 5, 12, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    merged = concatenate([x_1, x_2, x_3, x_4, x_5, x_6], axis=1)

    return input, merged

def model_layer2(input, merged, filter_density_layer2, dropout):

    # conv 2 layers
    merged = Conv2D(int(32 * filter_density_layer2), (3, 3), padding="valid",
                     data_format="channels_first",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5))(merged)
    # model_merged.add(BatchNormalization(axis=channel_axis, mode=0))

    merged = ELU()(merged)
    merged = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format="channels_first")(merged)
    merged = Dropout(dropout)(merged)
    merged = Flatten()(merged)

    # dense
    merged = Dense(units=256, kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5))(merged)
    merged = ELU()(merged)
    merged = Dropout(dropout)(merged)

    merged = Dense(1, activation='sigmoid')(merged)
    # model_1.add(Activation("softmax"))
    model_merged = Model(inputs=input, outputs=merged)

    optimizer = Adam()

    model_merged.compile(loss='binary_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])

    print(model_merged.summary())

    return model_merged

def model_layer2_schluter(input, merged, filter_density_layer2, dropout):
    """
    original schluter layer
    :param input:
    :param merged:
    :param filter_density_layer2:
    :param dropout:
    :return:
    """

    # conv 2 layers
    merged = Conv2D(int(20 * filter_density_layer2), (3, 3), padding="valid",
                     data_format="channels_first",
                     activation='relu')(merged)

    merged = MaxPooling2D(pool_size=(3, 1), padding='valid', data_format="channels_first")(merged)
    merged = Dropout(dropout)(merged)
    merged = Flatten()(merged)

    # dense
    merged = Dense(units=256, activation='sigmoid')(merged)
    merged = Dropout(dropout)(merged)

    merged = Dense(1, activation='sigmoid')(merged)
    model_merged = Model(inputs=input, outputs=merged)

    optimizer = Adam()

    model_merged.compile(loss='binary_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])

    print(model_merged.summary())

    return model_merged

def jordi_model(filter_density_1, filter_density_2,
                pool_n_row, pool_n_col,
                dropout, input_shape,
                dim='timbral'):
    if dim=='timbral':
        inputs, merged = timbral_layer(filter_density_layer1=filter_density_1,
                                       pool_n_row=pool_n_row,
                                       pool_n_col=pool_n_col,
                                       dropout=dropout,
                                       input_dim=input_shape)
    else:
        inputs, merged = temporal_layer(filter_density_layer1=filter_density_1,
                                       pool_n_row=pool_n_row,
                                       pool_n_col=pool_n_col,
                                       dropout=dropout,
                                       input_dim=input_shape)
    model = model_layer2(input=inputs,
                          merged=merged,
                          filter_density_layer2=filter_density_2,
                          dropout=dropout)

    return model

def jordi_model_schluter(filter_density_1, filter_density_2,
                        pool_n_row, pool_n_col,
                        dropout, input_shape,
                        dim='timbral'):
    """
    Schluter model configuration
    :param filter_density_1:
    :param filter_density_2:
    :param pool_n_row:
    :param pool_n_col:
    :param dropout:
    :param input_shape:
    :param dim:
    :return:
    """
    if dim=='timbral':
        inputs, merged = timbral_layer_schluter_valid(filter_density_layer1=filter_density_1,
                                                       pool_n_row=pool_n_row,
                                                       pool_n_col=pool_n_col,
                                                       dropout=dropout,
                                                       input_dim=input_shape)
    else:
        inputs, merged = temporal_layer_schluter(filter_density_layer1=filter_density_1,
                                               pool_n_row=pool_n_row,
                                               pool_n_col=pool_n_col,
                                               dropout=dropout,
                                               input_dim=input_shape)
    model = model_layer2_schluter(input=inputs,
                                  merged=merged,
                                  filter_density_layer2=filter_density_2,
                                  dropout=dropout)

    return model

def model_train(model_0, batch_size, patience, input_shape,
                path_feature_data,
                indices_train, Y_train, sample_weights_train,
                indices_validation, Y_validation, sample_weights_validation,
                indices_all, Y_train_validation, sample_weights, class_weights,
                file_path_model, filename_log):

    """
    train the model with validation early stopping and retrain the model with whole training dataset
    :param model_0:
    :param batch_size:
    :param patience:
    :param input_shape:
    :param path_feature_data:
    :param indices_train:
    :param Y_train:
    :param sample_weights_train:
    :param indices_validation:
    :param Y_validation:
    :param sample_weights_validation:
    :param indices_all:
    :param Y_train_validation:
    :param sample_weights:
    :param class_weights:
    :param file_path_model:
    :param filename_log:
    :return:
    """


    model_0.save_weights(basename(file_path_model))

    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]
    print("start training...")

    steps_per_epoch_train = int(np.ceil(len(indices_train) / batch_size))
    steps_per_epoch_val = int(np.ceil(len(indices_validation) / batch_size))

    generator_train = generator(path_feature_data=path_feature_data,
                                indices=indices_train,
                                number_of_batches=steps_per_epoch_train,
                                file_size=batch_size,
                                input_shape=input_shape,
                                labels=Y_train,
                                sample_weights=sample_weights_train,
                                multi_inputs=False)
    generator_val = generator(path_feature_data=path_feature_data,
                              indices=indices_validation,
                              number_of_batches=steps_per_epoch_val,
                              file_size=batch_size,
                              input_shape=input_shape,
                              labels=Y_validation,
                              sample_weights=sample_weights_validation,
                              multi_inputs=False)

    history = model_0.fit_generator(generator=generator_train,
                                    steps_per_epoch=steps_per_epoch_train,
                                    epochs=500,
                                    validation_data=generator_val,
                                    validation_steps=steps_per_epoch_val,
                                    class_weight=class_weights,
                                    callbacks=callbacks,
                                    verbose=2)

    model_0.load_weights(basename(file_path_model))

    # train again use all train and validation set
    epochs_final = len(history.history['val_loss'])
    # epochs_final = 100

    steps_per_epoch_train_val = int(np.ceil(len(indices_all) / batch_size))

    generator_train_val = generator(path_feature_data=path_feature_data,
                                    indices=indices_all,
                                    number_of_batches=steps_per_epoch_train_val,
                                    file_size=batch_size,
                                    input_shape=input_shape,
                                    labels=Y_train_validation,
                                    sample_weights=sample_weights,
                                    multi_inputs=False)

    model_0.fit_generator(generator=generator_train_val,
                          steps_per_epoch=steps_per_epoch_train_val,
                          epochs=epochs_final,
                          class_weight=class_weights,
                          verbose=2)

    model_0.save(file_path_model)
    remove(basename(file_path_model))


class MomentumScheduler(Callback):
    '''Momentum scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            momentum as output (float).
    '''
    def __init__(self, schedule):
        super(MomentumScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'momentum'), \
            'Optimizer must have a "momentum" attribute.'
        mmtm = self.schedule(epoch)
        assert type(mmtm) == float, 'The output of the "schedule" function should be float.'
        K.set_value(self.model.optimizer.momentum, mmtm)


def momentumIncrease(epoch):
    """
    increase momentum linearly from 0.45 to 0.9 between epoch 10 and 20
    :param epoch:
    :return:
    """
    if epoch <= 9:
        mmtm = 0.45
    elif epoch > 9 and epoch < 20:
        mmtm = 0.45 + 0.045*(epoch-9)
    else:
        mmtm = 0.9

    # print('epoch:', epoch, 'momentum:', mmtm)

    return mmtm

def lrDecrease(epoch):
    """
    decrease learning rate each epoch by a coefficient 0.995
    :param epoch:
    :return:
    """
    lr = np.power(0.995, epoch)
    # print('epoch:', epoch, 'Learning rate:', lr)
    return lr

def model_train_schluter(model_0,
                         batch_size,
                         input_shape,
                        path_feature_data,
                        indices_all,
                         Y_train_validation,
                         sample_weights,
                         class_weights,
                        file_path_model,
                         filename_log,
                         channel):

    # mmtm = MomentumScheduler(momentumIncrease)
    # lrSchedule = LearningRateScheduler(lrDecrease)
    # callbacks = [mmtm, lrSchedule, CSVLogger(filename=filename_log, separator=';')]
    callbacks = [CSVLogger(filename=filename_log, separator=';')]
    print("start training...")

    # train again use all train and validation set
    epochs_final = 100

    steps_per_epoch_train_val = int(np.ceil(len(indices_all) / batch_size))

    generator_train_val = generator(path_feature_data=path_feature_data,
                                    indices=indices_all,
                                    number_of_batches=steps_per_epoch_train_val,
                                    file_size=batch_size,
                                    input_shape=input_shape,
                                    labels=Y_train_validation,
                                    sample_weights=sample_weights,
                                    multi_inputs=False,
                                    channel=channel)

    model_0.fit_generator(generator=generator_train_val,
                          steps_per_epoch=steps_per_epoch_train_val,
                          epochs=epochs_final,
                          callbacks=callbacks,
                          # class_weight=class_weights,
                          verbose=2)

    model_0.save(file_path_model)
    # remove(basename(file_path_model))

def model_train_validation(model_0,
                           batch_size,
                           patience,
                           input_shape,
                            path_feature_data,
                            indices_train,
                           Y_train,
                           sample_weights_train,
                            indices_validation,
                           Y_validation,
                           sample_weights_validation,
                           class_weights,
                            file_path_model,
                           filename_log,
                           channel):

    """
    train the model with validation early stopping and retrain the model with whole training dataset
    :param model_0:
    :param batch_size:
    :param patience:
    :param input_shape:
    :param path_feature_data:
    :param indices_train:
    :param Y_train:
    :param sample_weights_train:
    :param indices_validation:
    :param Y_validation:
    :param sample_weights_validation:
    :param indices_all:
    :param Y_train_validation:
    :param sample_weights:
    :param class_weights:
    :param file_path_model:
    :param filename_log:
    :return:
    """

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_loss', verbose=0, save_best_only=True),
                EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]
    print("start training with validation...")

    steps_per_epoch_train = int(np.ceil(len(indices_train) / batch_size))
    steps_per_epoch_val = int(np.ceil(len(indices_validation) / batch_size))

    generator_train = generator(path_feature_data=path_feature_data,
                                indices=indices_train,
                                number_of_batches=steps_per_epoch_train,
                                file_size=batch_size,
                                input_shape=input_shape,
                                labels=Y_train,
                                sample_weights=sample_weights_train,
                                multi_inputs=False,
                                channel=channel)
    generator_val = generator(path_feature_data=path_feature_data,
                              indices=indices_validation,
                              number_of_batches=steps_per_epoch_val,
                              file_size=batch_size,
                              input_shape=input_shape,
                              labels=Y_validation,
                              sample_weights=sample_weights_validation,
                              multi_inputs=False,
                              channel=channel)

    model_0.fit_generator(generator=generator_train,
                            steps_per_epoch=steps_per_epoch_train,
                            epochs=500,
                            validation_data=generator_val,
                            validation_steps=steps_per_epoch_val,
                            callbacks=callbacks,
                            verbose=2)

def model_switcher(model_name,
                   filter_density,
                   dropout,
                   input_shape,
                   channel):
    if model_name == 'jan_original':
        model_0 = jan_original(filter_density=filter_density,
                               dropout=dropout,
                               input_shape=input_shape,
                               batchNorm=False,
                               dense_activation='sigmoid',
                               channel=channel)
    elif model_name == 'jordi_timbral_schluter':
        model_0 = jordi_model_schluter(filter_density_1=1,
                                       filter_density_2=filter_density,
                                       pool_n_row=11, # old 5
                                       pool_n_col=1, # old 3
                                       dropout=dropout,
                                       input_shape=input_shape,
                                       dim='timbral')
    elif model_name == 'jordi_temporal_schluter':
        model_0 = jordi_model_schluter(filter_density_1=2,
                                       filter_density_2=filter_density,
                                       pool_n_row=5, # old 3
                                       pool_n_col=1, # old 5
                                       dropout=dropout,
                                       input_shape=input_shape,
                                       dim='temporal')

    return model_0

def train_model(filename_train_validation_set,
                filename_labels_train_validation_set,
                filename_sample_weights,
                filter_density,
                dropout,
                input_shape,
                file_path_model,
                filename_log,
                model_name = 'jan_original',
                channel=1):
    """
    train final model save to model path
    """

    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data_schluter(filename_labels_train_validation_set,
                            filename_sample_weights)

    model_0 = model_switcher(model_name,filter_density,dropout,input_shape,channel)

    batch_size = 256

    # print(model_0.count_params())

    model_train_schluter(model_0,
                         batch_size,
                         input_shape,
                        filename_train_validation_set,
                        filenames_features,
                         Y_train_validation,
                         sample_weights,
                         class_weights,
                        file_path_model,
                         filename_log,
                         channel)

def train_model_validation(filename_train_validation_set,
                            filename_labels_train_validation_set,
                            filename_sample_weights,
                            filter_density,
                            dropout,
                            input_shape,
                            file_path_model,
                            filename_log,
                            model_name = 'jan_original',
                            channel=1):
    """
    train model with validation
    """

    filenames_train, Y_train, sample_weights_train, \
    filenames_validation, Y_validation, sample_weights_validation, \
    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data(filename_labels_train_validation_set,
                  filename_sample_weights)

    model_0 = model_switcher(model_name,filter_density,dropout,input_shape,channel)

    # print(model_0.summary())

    batch_size = 256
    patience = 15

    # print(model_0.count_params())

    model_train_validation(model_0,
                           batch_size,
                           patience,
                           input_shape,
                            filename_train_validation_set,
                            filenames_train, Y_train, sample_weights_train,
                            filenames_validation, Y_validation, sample_weights_validation,
                            class_weights,
                            file_path_model,
                           filename_log,
                           channel)
