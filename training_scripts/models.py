from os import remove
from os.path import basename

import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, Dense, Flatten, ELU
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2

from feature_generator import generator


def jan(filter_density, dropout, input_shape):

    reshape_dim = (1, input_shape[0], input_shape[1])

    model_1 = Sequential()
    model_1.add(Conv2D(int(10 * filter_density), (3, 7), padding="valid",
                    input_shape=reshape_dim, data_format="channels_first",
                     kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(3, 1), padding='valid',data_format="channels_first"))

    model_1.add(Conv2D(int(20 * filter_density), (3, 3), padding="valid", data_format="channels_first",
                       kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(3, 1), padding='valid',data_format="channels_first"))

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

def model_train(model_0, batch_size, patience, input_shape,
                path_feature_data,
                indices_train, Y_train, sample_weights_train,
                indices_validation, Y_validation, sample_weights_validation,
                indices_all, Y_train_validation, sample_weights, class_weights,
                file_path_model, filename_log):


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
