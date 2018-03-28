from file_path_jingju_shared import *
from os.path import join
from parameters_jingju import varin

if varin['architecture'] == 'jan_no_dense':
    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_madmom_early_stopping_no_dense'
    cnnModel_name = 'jan_artist_filter_madmom_early_stopping_no_dense'
elif varin['architecture'] == 'jan_sigmoid':
    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_artist_filter_madmom_early_stopping'
    cnnModel_name = 'jan_old+new_artist_filter_madmom_early_stopping'
elif varin['architecture'] == 'jan_relu':
    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_madmom_early_stopping_relu'
    cnnModel_name = 'jan_artist_filter_madmom_early_stopping_relu'
elif varin['architecture'] == 'jan_deep':
    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_madmom_early_stopping_deep'
    cnnModel_name = 'jan_artist_filter_madmom_early_stopping_deep'
elif varin['architecture'] == 'jan_less_deep':
    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_less_deep'
    cnnModel_name = 'jan_artist_filter_less_deep'
elif varin['architecture'] == 'jordi_temporal':
    filename_keras_cnn_0 = \
        'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_artist_filter_madmom_early_stopping_jan_params'
    cnnModel_name = 'jordi_temporal_artist_filter_madmom_early_stopping_jan_params'
elif varin['architecture'] == 'transfer_deep_feature':
    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_less_deep_deep_feature_extraction_schluter'
    cnnModel_name = 'jan_artist_filter_less_deep_deep_feature_extraction_schluter'
elif varin['architecture'] == 'transfer_2_layers_feature':
    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_less_deep_feature_extraction_schluter'
    cnnModel_name = 'jan_artist_filter_less_deep_feature_extraction_schluter'
elif varin['architecture'] == 'transfer_pretrained':
    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_less_deep_pretrained_schluter'
    cnnModel_name = 'jan_artist_filter_less_deep_pretrained_schluter'
elif varin['architecture'] == 'jan_less_deep_schluter':
    filename_keras_cnn_0 = 'schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_less_deep_'
    cnnModel_name = 'jan_artist_filter_less_deep_schluter'
else:
    raise ValueError('The architecture %s that you select is not a valid one.' % varin['architecture'])

# where we have the dumped features

if 'schluter' in varin['architecture']:
    cnnModels_path = join(root_path, 'cnnModels', 'schluter')
else:
    cnnModels_path = join(root_path, 'cnnModels', 'jingju')

if 'joint' in filename_keras_cnn_0:
    filename_scaler_onset = 'scaler_joint_subset.pkl'
elif 'schluter' in varin['architecture']:
    filename_scaler_onset = 'scaler_jan_madmom_simpleSampleWeighting_early_stopping_'
else:
    filename_scaler_onset = 'scaler_syllable_mfccBands2D_old+new_artist_filter_madmom.pkl'

full_path_keras_cnn_0 = join(cnnModels_path, varin['sample_weighting'], filename_keras_cnn_0)

full_path_mfccBands_2D_scaler_onset = join(cnnModels_path, varin['sample_weighting'], filename_scaler_onset)

eval_results_path = join(root_path, 'eval', 'results', cnnModel_name)

jingju_results_path = join(root_path, 'eval', 'jingju', 'results')
