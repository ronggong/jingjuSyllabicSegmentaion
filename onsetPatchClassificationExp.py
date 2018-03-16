"""
code to evaluate binary classification models
"""

import cPickle
import pickle
import gzip
import os
import h5py
import numpy as np
from src.utilFunctions import featureReshape
from experiment_process_jingju_no_rnn import late_fusion_calc
from keras.models import load_model
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def featureProcessing(feature, scaler, nlen):
    feature = np.array(feature, dtype='float32')
    feature_scaled = scaler.transform(feature)
    feature_reshaped = featureReshape(feature_scaled, nlen=nlen)
    return feature_reshaped

def getObs(filename_model, scaler, feature, model_flag='jan', expand_dim=True, nlen=10):
    model = load_model(os.path.join('./cnnModels/',filename_model))
    feature_processed = featureProcessing(feature, scaler, nlen)
    if expand_dim:
        feature_processed = np.expand_dims(feature_processed, axis=1)
    if model_flag == 'jan':
        obs = model.predict_proba(feature_processed, batch_size=128, verbose=2)
    else:
        obs = model.predict(feature_processed, batch_size=128, verbose=2)
    obs_0 = obs[:, 0]
    return obs_0

def getObsOld(filename_model, scaler, feature, model_flag='jan', nlen=10):
    model = load_model(os.path.join('./cnnModels/',filename_model))
    observations = featureProcessing(feature, scaler, nlen)
    if model_flag=='jordi':
        observations = [observations, observations, observations, observations, observations, observations]
    obs = model.predict_proba(observations, batch_size=128, verbose=2)
    obs_0 = obs[:, 1]
    return obs_0

def eval_metrics(y_pred, y_test):
    # print(classification_report(y_test, y_pred))
    # print confusion_matrix(y_test, y_pred)
    print("kappa score:")
    print(cohen_kappa_score(y_test, y_pred))
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    return confusion_matrix(y_test, y_pred)


def predictionResults(y_pred, y_test):
    AP = average_precision_score(y_test, y_pred)
    print("AUC precision-recall curve:")
    print(AP)
    y_pred_binary = [1 if p>0.5 else 0 for p in y_pred]
    conf_m = eval_metrics(y_pred_binary, y_test)
    return conf_m

def savePredictionResults(y_pred, label='ismir'):
    np.save(os.path.join('./eval/prediction_results/', label+'.npy'), y_pred)

def loadPredictionResults(label='ismir'):
    y_pred = np.load(os.path.join('./eval/prediction_results/', label + '.npy'))
    return y_pred

if __name__ == '__main__':
    # ismir madmom test
    from src.file_path_jingju_no_rnn import *
    filename_test_feature = join(feature_data_path, 'feature_all_ismir_madmom_test.h5')
    f = h5py.File(filename_test_feature, 'r')
    X_test = f['feature_all']

    filename_test_label = 'trainingData/labels_test_set_all_syllableSeg_mfccBands2D_old+new_ismir_madmom.pickle.gz'
    with gzip.open(filename_test_label, 'rb') as f:
        Y_test = cPickle.load(f)

    filename_scaler_oldnew_ismir_madmom_set = './cnnModels/scaler_syllable_mfccBands2D_old+new_ismir_madmom.pkl'
    scaler_oldnew_ismir_madmom_set = pickle.load(open(filename_scaler_oldnew_ismir_madmom_set, 'rb'))

    missing_onset_jan = []
    missing_onset_jordi = []
    peaks_jan = []
    peaks_jordi = []

    y_pred_jan = []
    y_pred_jordi = []
    y_test_all = []

    entire_onset = 0
    for ii in range(5):
        filename_jan_oldnew_ismir_madmom_model = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_ismir_madmom_early_stopping'+str(ii)+'.h5'
        filename_jordi_oldnew_temporal_ismir_madmom_model = 'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_ismir_madmom_early_stopping'+str(ii)+'.h5'

        y_pred_jan_oldnew_set = getObs(filename_jan_oldnew_ismir_madmom_model, scaler_oldnew_ismir_madmom_set, X_test, model_flag='jordi',nlen=7)
        print('jan oldnew set ismir madmom results:', ii)
        y_pred_jan += y_pred_jan_oldnew_set.tolist()
        # conf_m = predictionResults(y_pred_jan_oldnew_set, Y_test)
        # missing_onset_jan.append(conf_m[1][0])
        # peaks_jan += [y for y in y_pred_jan_oldnew_set if y > 0.1]

        y_pred_jordi_temporal_oldnew_set = getObs(filename_jordi_oldnew_temporal_ismir_madmom_model, scaler_oldnew_ismir_madmom_set, X_test, model_flag='jordi',nlen=7)
        print('jordi temporal oldnew set ismir madmom results:', ii)
        y_pred_jordi += y_pred_jordi_temporal_oldnew_set.tolist()
        # conf_m = predictionResults(y_pred_jordi_temporal_oldnew_set, Y_test)
        # missing_onset_jordi.append(conf_m[1][0])
        # peaks_jordi += [y for y in y_pred_jordi_temporal_oldnew_set if y>0.1]

        y_test_all += Y_test.tolist()
        # entire_onset = sum(conf_m[1,:])

    conf_m_jan = predictionResults(y_pred_jan, y_test_all)
    missing_onset_jan.append(conf_m_jan[1][0])
    peaks_jan += [y for y in y_pred_jan if y > 0.1]

    conf_m_jordi = predictionResults(y_pred_jordi, y_test_all)
    missing_onset_jordi.append(conf_m_jordi[1][0])
    peaks_jordi += [y for y in y_pred_jordi if y > 0.1]

    print(np.mean(missing_onset_jan) / float(entire_onset))
    print(np.mean(missing_onset_jordi) / float(entire_onset))

    print(np.mean(peaks_jan), np.std(peaks_jan))
    print(np.mean(peaks_jordi), np.std(peaks_jordi))


    # filename_test_feature = 'trainingData/feature_test_set_all_syllableSeg_mfccBands2D_old+new_artist_split.h5'
    # f = h5py.File(filename_test_feature, 'r')
    # X_test = f['feature_all']
    #
    # filename_test_label = 'trainingData/label_test_set_all_syllableSeg_mfccBands2D_old+new_artist_split.pickle.gz'
    # with gzip.open(filename_test_label, 'rb') as f:
    #     Y_test = cPickle.load(f)

    # filename_scaler_oldnew_artist_split_set = './cnnModels/scaler_syllable_mfccBands2D_old+new_artist_split.pkl'
    # scaler_oldnew_artist_split_set = pickle.load(open(filename_scaler_oldnew_artist_split_set, 'rb'))
    #
    # # artist splits
    # filename_jan_oldnew_artist_split_model = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_artist_split.h5'
    # filename_jan_deep_oldnew_artist_split_model = 'keras.cnn_syllableSeg_jan_deep_class_weight_mfccBands_2D_all_artist_split.h5'
    # filename_jordi_oldnew_temporal_artist_split_model = 'keras.cnn_syllableSeg_jordi_temporal_class_weight_with_conv_dense_mfccBands_2D_artist_split.h5'
    # filename_jordi_oldnew_timbral_artist_split_model = 'keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_artist_split.h5'
    #
    # # artist split
    # y_pred_jan_oldnew_set = getObs(filename_jan_oldnew_artist_split_model, scaler_oldnew_artist_split_set, X_test, model_flag='jan')
    # print('jan oldnew set artist split results:')
    # predictionResults(y_pred_jan_oldnew_set, Y_test)
    # savePredictionResults(y_pred_jan_oldnew_set, label='jan_artist_split')
    #
    # y_pred_jan_deep_oldnew_set = getObs(filename_jan_deep_oldnew_artist_split_model, scaler_oldnew_artist_split_set, X_test, model_flag='jan')
    # print('jan deep oldnew set artist split results:')
    # predictionResults(y_pred_jan_deep_oldnew_set, Y_test)
    # savePredictionResults(y_pred_jan_deep_oldnew_set, label='jan_deep_artist_split')
    #
    # y_pred_jordi_temporal_oldnew_set = getObs(filename_jordi_oldnew_temporal_artist_split_model, scaler_oldnew_artist_split_set, X_test, model_flag='jordi')
    # print('jordi temporal oldnew set artist split results:')
    # predictionResults(y_pred_jordi_temporal_oldnew_set, Y_test)
    # savePredictionResults(y_pred_jordi_temporal_oldnew_set, label='temporal_artist_split')
    #
    # y_pred_jordi_timbral_oldnew_set = getObs(filename_jordi_oldnew_timbral_artist_split_model, scaler_oldnew_artist_split_set, X_test, model_flag='jordi')
    # print('jordi timbral oldnew set artist split results:')
    # predictionResults(y_pred_jordi_timbral_oldnew_set, Y_test)
    # savePredictionResults(y_pred_jordi_timbral_oldnew_set, label='timbral_artist_split')

    # y_pred_jordi_timbral_oldnew_set = loadPredictionResults(label = 'timbral_artist_split')
    # y_pred_jordi_temporal_oldnew_set = loadPredictionResults(label = 'temporal_artist_split')
    # y_pred_jordi_fusion_oldnew_set = late_fusion_calc(y_pred_jordi_temporal_oldnew_set, y_pred_jordi_timbral_oldnew_set, mth=2, coef=0.5)
    # print('jordi fusion oldnew set artist split results:')
    # predictionResults(y_pred_jordi_fusion_oldnew_set, Y_test)

    # artist filter split

    # filename_test_feature = 'trainingData/feature_test_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_split.h5'
    # f = h5py.File(filename_test_feature, 'r')
    # X_test = f['feature_all']
    #
    # filename_test_label = 'trainingData/label_test_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_split.pickle.gz'
    # with gzip.open(filename_test_label, 'rb') as f:
    #     Y_test = cPickle.load(f)
    #
    # filename_scaler_oldnew_artist_filter_split_set = './cnnModels/scaler_syllable_mfccBands2D_old+new_artist_filter_split.pkl'
    # scaler_oldnew_artist_filter_split_set = pickle.load(open(filename_scaler_oldnew_artist_filter_split_set, 'rb'))
    #
    # filename_jan_oldnew_artist_filter_split_model = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_artist_filter_split.h5'
    # filename_jan_deep_oldnew_artist_filter_split_model = 'keras.cnn_syllableSeg_jan_deep_class_weight_mfccBands_2D_all_artist_filter_split.h5'
    # filename_jordi_oldnew_temporal_artist_filter_split_model = 'keras.cnn_syllableSeg_jordi_temporal_class_weight_with_conv_dense_mfccBands_2D_artist_filter_split.h5'
    # filename_jordi_oldnew_timbral_artist_filter_split_model = 'keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_artist_filter_split_2_train.h5'

    # # artist filter split
    # # y_pred_jan_oldnew_set = getObs(filename_jan_oldnew_artist_filter_split_model, scaler_oldnew_artist_filter_split_set, X_test, model_flag='jan')
    # y_pred_jan_oldnew_set = loadPredictionResults(label = 'jan_artist_filter_split')
    # print('jan oldnew set artist filter split results:')
    # predictionResults(y_pred_jan_oldnew_set, Y_test)
    # # savePredictionResults(y_pred_jan_oldnew_set, label='jan_artist_filter_split')
    #
    # y_pred_jan_deep_oldnew_set = getObs(filename_jan_deep_oldnew_artist_filter_split_model, scaler_oldnew_artist_filter_split_set, X_test, model_flag='jan')
    # y_pred_jan_deep_oldnew_set = loadPredictionResults(label = 'jan_deep_artist_filter_split')
    # print('jan deep oldnew set artist filter split results:')
    # predictionResults(y_pred_jan_deep_oldnew_set, Y_test)
    # savePredictionResults(y_pred_jan_deep_oldnew_set, label='jan_deep_artist_filter_split')
    #
    # # y_pred_jordi_temporal_oldnew_set = getObs(filename_jordi_oldnew_temporal_artist_filter_split_model, scaler_oldnew_artist_filter_split_set, X_test, model_flag='jordi')
    # y_pred_jordi_temporal_oldnew_set = loadPredictionResults(label = 'temporal_artist_filter_split')
    # print('jordi temporal oldnew set artist filter split results:')
    # predictionResults(y_pred_jordi_temporal_oldnew_set, Y_test)
    # # savePredictionResults(y_pred_jordi_temporal_oldnew_set, label='temporal_artist_filter_split')

    # y_pred_jordi_timbral_oldnew_set = getObs(filename_jordi_oldnew_timbral_artist_filter_split_model, scaler_oldnew_artist_filter_split_set, X_test, model_flag='jordi')
    # y_pred_jordi_timbral_oldnew_set = loadPredictionResults(label = 'timbral_artist_filter_split_2_train')
    # print('jordi timbral oldnew set artist filter split results:')
    # predictionResults(y_pred_jordi_timbral_oldnew_set, Y_test)
    # savePredictionResults(y_pred_jordi_timbral_oldnew_set, label='timbral_artist_filter_split_2_train')

    # y_pred_jordi_timbral_oldnew_set = loadPredictionResults(label = 'timbral_artist_filter_split_2_train')
    # y_pred_jordi_temporal_oldnew_set = loadPredictionResults(label = 'temporal_artist_filter_split')
    # y_pred_jordi_fusion_oldnew_set = late_fusion_calc(y_pred_jordi_temporal_oldnew_set, y_pred_jordi_timbral_oldnew_set, mth=2, coef=0.5)
    # print('jordi fusion oldnew set artist filter split results:')
    # predictionResults(y_pred_jordi_fusion_oldnew_set, Y_test)