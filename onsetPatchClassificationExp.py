"""
code to evaluate binary classification models
"""

import cPickle
import pickle
import gzip
import os
import numpy as np
from trainingSampleCollection import featureReshape
from keras.models import load_model
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

def featureProcessing(feature, scaler):
    feature = np.array(feature, dtype='float32')
    feature_scaled = scaler.transform(feature)
    feature_reshaped = featureReshape(feature_scaled)
    return feature_reshaped

def getObs(filename_model, scaler, feature, model_flag='jan', expand_dim=True):
    model = load_model(os.path.join('./cnnModels/',filename_model))
    feature_processed = featureProcessing(feature, scaler)
    if expand_dim:
        feature_processed = np.expand_dims(feature_processed, axis=1)
    if model_flag == 'jan':
        obs = model.predict_proba(feature_processed, batch_size=128, verbose=2)
    else:
        obs = model.predict(feature_processed, batch_size=128, verbose=2)
    obs_0 = obs[:, 0]
    return obs_0

def getObsOld(filename_model, scaler, feature, model_flag='jan'):
    model = load_model(os.path.join('./cnnModels/',filename_model))
    observations = featureProcessing(feature, scaler)
    if model_flag=='jordi':
        observations = [observations, observations, observations, observations, observations, observations]
    obs = model.predict_proba(observations, batch_size=128, verbose=2)
    obs_0 = obs[:, 1]
    return obs_0

def eval_metrics(y_pred, y_test):
    # print(classification_report(y_test, y_pred))
    # print confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:")
    print(accuracy)
    print("Micro stats:")
    print(precision_recall_fscore_support(y_test, y_pred, average='micro'))
    print("Macro stats:")
    print(precision_recall_fscore_support(y_test, y_pred, average='macro'))

    return accuracy

def predictionResults(y_pred, y_test):
    roc_value = roc_auc_score(y_test, y_pred)
    print("roc auc:")
    print(roc_value)
    y_pred_binary = [1 if p>0.5 else 0 for p in y_pred]
    eval_metrics(y_pred_binary, y_test)


filename_test_set = './trainingData/test_set_all_syllableSeg_mfccBands2D.pickle.gz'
with gzip.open(filename_test_set, 'rb') as f:
    X_test, Y_test = cPickle.load(f)

filename_scaler_old_set = './cnnModels/scaler_syllable_mfccBands2D.pkl'
filename_scaler_oldnew_set = './cnnModels/scaler_syllable_mfccBands2D_old+new_ismir_split.pkl'

scaler_old_set = pickle.load(open(filename_scaler_old_set, 'rb'))
scaler_oldnew_set = pickle.load(open(filename_scaler_oldnew_set, 'rb'))

filename_jan_oldnew_model = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_ismir_split.h5'
filename_jan_deep_oldnew_model = 'keras.cnn_syllableSeg_jan_deep_class_weight_mfccBands_2D_all_ismir_split.h5'
filename_jordi_oldnew_temporal_model = 'keras.cnn_syllableSeg_jordi_temporal_class_weight_with_conv_dense_mfccBands_2D_ismir_split.h5'
filename_jordi_oldnew_timbral_model = 'keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_ismir_split.h5'

filename_jan_old_model = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_optim.h5'
filename_jordi_old_temporal_model = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_mfccBands_2D_all_optim.h5'
filename_jordi_old_timbral_model = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_timbral_filter_mfccBands_node04_2D_all_optim.h5'

# old + new dataset test

# y_pred_jordi_temporal_oldnew_set = getObs(filename_jordi_oldnew_temporal_model, scaler_oldnew_set, X_test, model_flag='jordi')
# print('jordi temporal oldnew set results:')
# predictionResults(y_pred_jordi_temporal_oldnew_set, Y_test)
#
# y_pred_jordi_timbral_oldnew_set = getObs(filename_jordi_oldnew_timbral_model, scaler_oldnew_set, X_test, model_flag='jordi')
# print('jordi timbral oldnew set results:')
# predictionResults(y_pred_jordi_timbral_oldnew_set, Y_test)
#
# y_pred_jan_oldnew_set = getObs(filename_jan_oldnew_model, scaler_oldnew_set, X_test, model_flag='jan')
# print('jan oldnew set results:')
# predictionResults(y_pred_jan_oldnew_set, Y_test)
#
# y_pred_jan_deep_oldnew_set = getObs(filename_jan_deep_oldnew_model, scaler_oldnew_set, X_test, model_flag='jan')
# print('jan deep oldnew set results:')
# predictionResults(y_pred_jan_deep_oldnew_set, Y_test)


y_pred_jan_oldnew_set = getObsOld(filename_jan_old_model, scaler_old_set, X_test, model_flag='jan')
print('jan old set results:')
predictionResults(y_pred_jan_oldnew_set, Y_test)

y_pred_jordi_temporal_oldnew_set = getObsOld(filename_jordi_old_temporal_model, scaler_old_set, X_test, model_flag='jordi')
print('jordi temporal old set results:')
predictionResults(y_pred_jordi_temporal_oldnew_set, Y_test)

y_pred_jordi_timbral_oldnew_set = getObsOld(filename_jordi_old_timbral_model, scaler_old_set, X_test, model_flag='jordi')
print('jordi timbral old set results:')
predictionResults(y_pred_jordi_timbral_oldnew_set, Y_test)