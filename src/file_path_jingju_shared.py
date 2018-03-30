from os.path import join
from os.path import dirname
from parameters_jingju import varin
from file_path_shared import feature_data_path

#  audio and annotation root path
root_path = join(dirname(__file__), '..')

# nacta 2017 dataset part 2
# nacta2017_dataset_root_path = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset_extended_nacta2017'
nacta2017_dataset_root_path = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/jingju_a_cappella_singing_dataset_extended_nacta2017'

# nacta dataset part 1
# nacta_dataset_root_path = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset'
nacta_dataset_root_path = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/jingju_a_cappella_singing_dataset'

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

jingju_cnn_model_path = join(root_path, 'cnnModels', 'jingju', varin['sample_weighting'])

jingju_scaler_path = join(root_path, 'cnnModels', 'jingju', 'scalers')

cnnModels_path = join(root_path, 'cnnModels', 'jingju')

artist_filter_feature_data_path = join(feature_data_path, 'artist_filter')
