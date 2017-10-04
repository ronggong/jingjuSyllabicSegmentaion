from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import cPickle
import gzip
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np


# load training and validation data
filename_train_validation_set = '/scratch/rgongcnnSyllableSeg_jan_deep/syllableSeg/train_set_all_syllableSeg_mfccBands2D.pickle.gz'
filename_sample_weights = '/scratch/rgongcnnSyllableSeg_jan_deep/syllableSeg/sample_weights_syllableSeg_mfccBands2D.pickle.gz'

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
X_train                                         = np.array([xt[0] for xt in X_train], dtype='float32')
X_validation                                    = np.array([xv[0] for xv in X_validation], dtype='float32')

x_train = np.expand_dims(X_train, axis=1)
x_validation = np.expand_dims(X_validation, axis=1)
x_train_validation = np.expand_dims(X_train_validation, axis=1)

Y_train_validation                              = to_categorical(Y_train_validation)
Y_train                                         = to_categorical(Y_train)
Y_validation                                    = to_categorical(Y_validation)

print(sample_weights_X_train.shape, X_train.shape, Y_train.shape)
print(X_validation.shape, Y_validation.shape)


from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten, ELU
from keras.regularizers import l2


def f_nn_model(filter_density, dropout):

    nlen = 21
    reshape_dim = (1, 80, nlen)
    # input_dim = (80, nlen, 3)
    # input_dim = (80, nlen)

    # reshape_dim = (1, input_shape[0], input_shape[1])

    model_1 = Sequential()
    model_1.add(Conv2D(int(16 * filter_density), (3, 7), padding="valid",
                       input_shape=reshape_dim, data_format="channels_first",
                       kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    # model_1.add(MaxPooling2D(pool_size=(2, 2), padding='valid',data_format="channels_first"))

    model_1.add(Conv2D(int(16 * filter_density), (3, 3), padding="valid", data_format="channels_first",
                       kernel_initializer="he_uniform", kernel_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(3, 2), padding='valid', data_format="channels_first"))

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

    model_1.add(Dense(units=2, activation="softmax"))

    # optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
    optimizer = Adam()

    model_1.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

    print(model_1.summary())

    return model_1


def train_model(filter_density, dropout, file_path_model):
    """
    train final model save to model path
    """
    model_0 = f_nn_model(filter_density, dropout)

    print(model_0.count_params())

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]
    print("start training...")
    hist = model_0.fit(X_train,
                            Y_train,
                            validation_data=[X_validation, Y_validation],
                            class_weight=class_weights,
                            callbacks=callbacks,
                            sample_weight=sample_weights_X_train,
                            epochs=500,
                            batch_size=128)
    nb_epoch = len(hist.history['acc'])

    model_merged_1 = f_nn_model(filter_density, dropout)

    model_merged_1.fit(X_train_validation,
                                Y_train_validation,
                              class_weight=class_weights,
                              sample_weight=sample_weights,
                                epochs=nb_epoch,
                                batch_size=128)

    model_merged_1.save(file_path_model)


if __name__ == '__main__':


    # train the final model
    file_path_model = '/scratch/rgongcnnSyllableSeg_jan_deep/out/keras.cnn_syllableSeg_jan_deep_class_weight_mfccBands_2D_all_old_ismir.h5'
    train_model(filter_density=1, dropout=0.5, file_path_model=file_path_model)
