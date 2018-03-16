from file_path_jingju_shared import *
from os.path import join
from parameters_jingju import varin

# where we have the dumped features
feature_data_path = '/Users/gong/Documents/MTG document/dataset/syllableSeg/'

cnnModels_path  = join(root_path, 'cnnModels', 'jingju')
# TODO use schluter to evaluate jingju
# cnnModels_path = join(root_path, 'cnnModels', 'schluter')

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

    # TODO use schluter to evaluate jingju
    # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_less_deep_deep_feature_extraction_schluter'
    # cnnModel_name = 'jan_artist_filter_less_deep_deep_feature_extraction_schluter'

elif varin['architecture'] == 'jordi_temporal':
    filename_keras_cnn_0 = \
        'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_artist_filter_madmom_early_stopping_jan_params'
    cnnModel_name = 'jordi_temporal_artist_filter_madmom_early_stopping_jan_params'
else:
    raise ValueError('The architecture %s that you select is not a valid one.', varin['architecture'])


if 'joint' in filename_keras_cnn_0:
    filename_scaler_onset = 'scaler_joint_subset.pkl'
else:
    # TODO use schluter to evaluate jingju
    filename_scaler_onset = 'scaler_syllable_mfccBands2D_old+new_artist_filter_madmom.pkl'
    # filename_scaler_onset = 'scaler_jan_madmom_simpleSampleWeighting_early_stopping_schluter_jingju_dataset_'

full_path_keras_cnn_0 = join(cnnModels_path, varin['sample_weighting'], filename_keras_cnn_0)

full_path_mfccBands_2D_scaler_onset = join(cnnModels_path, varin['sample_weighting'], filename_scaler_onset)

eval_results_path = join(root_path, 'eval', 'results', cnnModel_name)

jingju_results_path = join(root_path, 'eval', 'jingju', 'results')
