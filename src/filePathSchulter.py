from os.path import join, dirname

schluter_dataset_root_path     = '/Users/gong/Documents/MTG document/dataset/onsets'

# schluter_dataset_root_path = '/datasets/MTG/projects/compmusic/jingju_datasets/schluter/'

schluter_audio_path = join(schluter_dataset_root_path, 'audio')
schluter_cv_path = join(schluter_dataset_root_path, 'splits')
schluter_annotations_path = join(schluter_dataset_root_path, 'annotations')

schluter_feature_data_path = join(schluter_dataset_root_path, 'feature')
schluter_feature_data_path_madmom = join(schluter_dataset_root_path, 'feature_madmom')
schluter_feature_data_path_madmom_simpleSampleWeighting = join(schluter_dataset_root_path, 'feature_madmom_simpleSampleWeighting')
schluter_feature_data_path_madmom_simpleSampleWeighting_3channel = join(schluter_dataset_root_path, 'feature_madmom_simpleSampleWeighting_3channel')

schluter_feature_data_path_madmom_positiveThreeSampleWeighting = join(schluter_dataset_root_path, 'feature_madmom_postiveThreeSampleWeighting')

schluter_feature_data_path_madmom_simpleSampleWeighting_phrase = join(schluter_dataset_root_path, 'feature_madmom_simpleSampleWeighting_phrase')

root_path       = join(dirname(__file__),'..')

weighting = 'simpleWeighting'

schluter_cnn_model_path = join(root_path, 'cnnModels', 'schluter', weighting)

scaler_schluter_phrase_model_path = join(schluter_dataset_root_path,
                                             'scaler_syllable_mfccBands2D_schluter_madmom_phrase.pkl')

eval_results_path = join(root_path, 'eval', 'results')

schluter_results_path = join(root_path, 'eval', 'schluter', 'results')

batch_size = 64
len_seq = 100  # sub-sequence length