import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import cPickle, pickle
import gzip
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

# load training and validation data
filename_train_validation_set = '/scratch/rgongcnnSyllableSeg_jan/syllableSeg/train_set_all_syllableSeg_onset_mfccBands2D_23.pickle.gz'
filename_train_validation_set_46 = '/scratch/rgongcnnSyllableSeg_jan/syllableSeg/train_set_all_syllableSeg_onset_mfccBands2D_46.pickle.gz'
filename_train_validation_set_93 = '/scratch/rgongcnnSyllableSeg_jan/syllableSeg/train_set_all_syllableSeg_onset_mfccBands2D_93.pickle.gz'

filename_sample_weights = '/scratch/rgongcnnSyllableSeg_jan/syllableSeg/sample_weights_syllableSeg_onset_mfccBands2D_23.pickle.gz'

with gzip.open(filename_train_validation_set, 'rb') as f:
    X_train_validation, Y_train_validation = cPickle.load(f)

with gzip.open(filename_train_validation_set_46, 'rb') as f:
    X_train_validation_46, Y_train_validation_46 = cPickle.load(f)

with gzip.open(filename_train_validation_set_93, 'rb') as f:
    X_train_validation_93, Y_train_validation_93 = cPickle.load(f)

with gzip.open(filename_sample_weights, 'rb') as f:
    sample_weights = cPickle.load(f)

X_train_validation = X_train_validation[...,np.newaxis]
X_train_validation_46 = X_train_validation_46[...,np.newaxis]
X_train_validation_93 = X_train_validation_93[...,np.newaxis]

X_train_validation = np.concatenate((X_train_validation, X_train_validation_46, X_train_validation_93), axis=3)

print(X_train_validation.shape, Y_train_validation.shape, sample_weights.shape)
print(len(Y_train_validation[Y_train_validation==0]), len(Y_train_validation[Y_train_validation==1]))
class_weights = compute_class_weight('balanced',[0,1],Y_train_validation)
print(class_weights)
class_weights = {0:class_weights[0], 1:class_weights[1]}

# packing
X_train_validation                              = np.array([[X_train_validation[ii], sample_weights[ii]] for ii in range(len(X_train_validation))])

X_train, X_validation, Y_train, Y_validation    = train_test_split(X_train_validation, Y_train_validation, test_size=0.2, stratify=Y_train_validation)

# unpacking
sample_weights_X_train                          = np.array([xt[1] for xt in X_train])
X_train_validation                              = np.array([xtv[0] for xtv in X_train_validation], dtype='float32')
X_train                                         = np.array([xt[0] for xt in X_train], dtype='float32')
X_validation                                    = np.array([xv[0] for xv in X_validation], dtype='float32')

Y_train_validation                              = to_categorical(Y_train_validation)
Y_train                                         = to_categorical(Y_train)
Y_validation                                    = to_categorical(Y_validation)


from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten, Merge, ELU
from keras.regularizers import l2


def f_nn_model(filter_density, dropout):

    nlen = 21
    # reshape_dim = (1, 80, nlen)
    input_dim = (80, nlen, 3)

    model_1 = Sequential()
    # model_1.add(Reshape(reshape_dim, input_shape=input_dim))
    model_1.add(Convolution2D(10 * filter_density, 3, 7, border_mode='valid', input_shape=input_dim, dim_ordering='tf',
                      init='he_uniform', W_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(3, 1), border_mode='valid',dim_ordering='tf'))

    model_1.add(Convolution2D(20 * filter_density, 3, 3, border_mode='valid', dim_ordering='tf',
                      init='he_uniform', W_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(3, 1), border_mode='valid', dim_ordering='tf'))

    model_1.add(Flatten())

    model_1.add(Dense(output_dim=256, init='he_uniform', W_regularizer=l2(1e-5)))
    model_1.add(ELU())
    model_1.add(Dropout(dropout))

    model_1.add(Dense(output_dim=2))
    model_1.add(Activation("softmax"))

    # optimizer = SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
    optimizer = Adam()

    model_1.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])

    return model_1


def train_model(filter_density, dropout, file_path_model):
    """
    train final model save to model path
    """
    model_merged_0 = f_nn_model(filter_density, dropout)

    print(model_merged_0.count_params())

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]
    print("start training...")
    hist = model_merged_0.fit(X_train,
                            Y_train,
                            validation_data=[X_validation,Y_validation],
                            class_weight=class_weights,
#                            validation_split=0.2,
                            callbacks=callbacks,
                            sample_weight=sample_weights_X_train,
                            nb_epoch=500,
                            batch_size=128)
                            
    nb_epoch = len(hist.history['acc'])

    model_merged_1 = f_nn_model(filter_density, dropout)

    hist = model_merged_1.fit(X_train_validation,
                                Y_train_validation,
                              class_weight=class_weights,
                              sample_weight=sample_weights,
                                nb_epoch=nb_epoch,
                                batch_size=128)

    model_merged_1.save(file_path_model)


if __name__ == '__main__':

    file_path_model = '/scratch/rgongcnnSyllableSeg_jan/out/keras.cnn_syllableSeg_jan_class_weight_3_chans_layer1_15_mfccBands_2D_all_optim.h5'
    train_model(filter_density=1, dropout=0.2503, file_path_model=file_path_model)
