from os.path import join, dirname
from parameters import *

jingju_dataset_root_path     = '/Users/gong/Documents/MTG document/dataset/syllableSeg/'

# jingju_dataset_root_path = '/homedtic/rgong/cnnSyllableSeg/syllableSeg'

ismir_feature_data_path = join(jingju_dataset_root_path, 'ismir')
artist_filter_feature_data_path = join(jingju_dataset_root_path, 'artist_filter')

root_path       = join(dirname(__file__),'..')

weighting = varin['sample_weighting']

jingju_cnn_model_path = join(root_path, 'cnnModels', 'jingju', weighting)

scaler_ismir_phrase_model_path = join(jingju_dataset_root_path,
                                    'scaler_syllable_mfccBands2D_old+new_ismir_madmom_phrase.pkl')
scaler_artist_filter_phrase_model_path = join(jingju_dataset_root_path,
                                        'scaler_syllable_mfccBands2D_old+new_artist_filter_madmom_phrase.pkl')

if varin['overlap']:
    cnnModel_name = varin['dataset']+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase_overlap_bottleneck_bidi'
else:
    cnnModel_name = varin['dataset']+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase_bidi'

eval_results_path = join(root_path, 'eval', 'results', cnnModel_name)

jingju_results_path = join(root_path, 'eval', 'jingju', 'results')

full_path_keras_cnn_0 = join(jingju_cnn_model_path, cnnModel_name)

batch_size = 64
len_seq = 400  # sub-sequence length

# nacta 2017 dataset part 2
nacta2017_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset_extended_nacta2017'

# nacta dataset part 1
nacta_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset'

nacta2017_wav_path = join(nacta2017_dataset_root_path, 'wav')
nacta2017_textgrid_path = join(nacta2017_dataset_root_path, 'textgridDetails')
nacta2017_score_path = join(nacta2017_dataset_root_path, 'scoreDianSilence')
nacta2017_score_pinyin_path = join(nacta2017_dataset_root_path, 'scoreDianSilence_pinyin')
nacta2017_score_pinyin_corrected_path = join(nacta2017_dataset_root_path, 'scoreDianSilence_pinyin_corrected')
nacta2017_segPhrase_path = join(nacta2017_dataset_root_path, 'segPhrase')
nacta2017_groundtruthlab_path = join(nacta2017_dataset_root_path, 'groundtruth_lab')
nacta2017_eval_details_path = join(nacta2017_dataset_root_path, 'eval_details')


nacta_wav_path = join(nacta_dataset_root_path, 'wav')
nacta_textgrid_path = join(nacta_dataset_root_path, 'textgrid')
nacta_score_path = join(nacta_dataset_root_path, 'scoreDianSilence')
nacta_score_pinyin_path = join(nacta_dataset_root_path, 'scoreDianSilence_pinyin')
nacta_score_pinyin_corrected_path = join(nacta_dataset_root_path, 'scoreDianSilence_pinyin_corrected')
# nacta_score_path = '/Users/gong/Documents/github/MTG/JingjuSingingAnnotation/aCapella/Syllable duration annotation'
nacta_segPhrase_path = join(nacta_dataset_root_path, 'segPhrase')
nacta_groundtruthlab_path = join(nacta_dataset_root_path, 'groundtruth_lab')
nacta_eval_details_path = join(nacta_dataset_root_path, 'eval_details')

# unified score path
if varin['corrected_score_duration']:
    nacta2017_score_unified_path = nacta2017_score_pinyin_corrected_path
    nacta_score_unified_path = nacta_score_pinyin_corrected_path
else:
    nacta2017_score_unified_path = nacta2017_score_pinyin_path
    nacta_score_unified_path = nacta_score_pinyin_path
