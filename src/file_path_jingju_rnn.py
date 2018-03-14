from file_path_jingju_shared import *
from os.path import join, dirname
from parameters import *

weighting = varin['sample_weighting']

jingju_dataset_root_path = '/Users/gong/Documents/MTG document/dataset/syllableSeg/'

# jingju_dataset_root_path = '/homedtic/rgong/cnnSyllableSeg/syllableSeg'

ismir_feature_data_path = join(jingju_dataset_root_path, 'ismir')

artist_filter_feature_data_path = join(jingju_dataset_root_path, 'artist_filter')

jingju_cnn_model_path = join(root_path, 'cnnModels', 'jingju', weighting)

scaler_ismir_phrase_model_path = join(jingju_dataset_root_path,
                                    'scaler_syllable_mfccBands2D_old+new_ismir_madmom_phrase.pkl')
scaler_artist_filter_phrase_model_path = join(jingju_dataset_root_path,
                                        'scaler_syllable_mfccBands2D_old+new_artist_filter_madmom_phrase.pkl')

if varin['overlap']:
    cnnModel_name = varin['dataset']+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase_overlap_bidi_100'
else:
    cnnModel_name = varin['dataset']+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase'

eval_results_path = join(root_path, 'eval', 'results', cnnModel_name)

jingju_results_path = join(root_path, 'eval', 'jingju', 'results')

full_path_keras_cnn_0 = join(jingju_cnn_model_path, cnnModel_name)

batch_size = 64
len_seq = 100  # sub-sequence length
