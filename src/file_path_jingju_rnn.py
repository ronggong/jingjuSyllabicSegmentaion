from file_path_shared import feature_data_path
from file_path_jingju_shared import *
from os.path import join
from parameters_jingju import *

if varin['architecture'] == 'jan_bidi_100':
    cnnModel_name = varin['dataset']+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase_overlap_bidi_100'
    len_seq = 100  # sub-sequence length
elif varin['architecture'] == 'jan_bidi_200':
    cnnModel_name = varin['dataset']+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase_overlap_bidi_200'
    len_seq = 200
elif varin['architecture'] == 'jan_bidi_400':
    cnnModel_name = varin['dataset']+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase_overlap_bidi'
    len_seq = 400
else:
    raise ValueError('There is no such architecture %s for CRNN.' % varin['architecture'])

ismir_feature_data_path = join(feature_data_path, 'ismir')

artist_filter_feature_data_path = join(feature_data_path, 'artist_filter')

scaler_ismir_phrase_model_path = join(jingju_scaler_path,
                                      'scaler_syllable_mfccBands2D_old+new_ismir_madmom_phrase.pkl')

scaler_artist_filter_phrase_model_path = join(jingju_scaler_path,
                                              'scaler_syllable_mfccBands2D_old+new_artist_filter_madmom_phrase.pkl')

eval_results_path = join(root_path, 'eval', 'results', cnnModel_name)

jingju_results_path = join(root_path, 'eval', 'jingju', 'results')

full_path_keras_cnn_0 = join(jingju_cnn_model_path, cnnModel_name)

# jingju_dataset_root_path = '/Users/gong/Documents/MTG document/dataset/syllableSeg/'

# jingju_dataset_root_path = '/homedtic/rgong/cnnSyllableSeg/syllableSeg'
