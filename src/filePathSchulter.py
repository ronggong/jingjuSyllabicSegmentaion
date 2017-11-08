from os.path import join

# schluter_dataset_root_path     = '/Users/gong/Documents/MTG document/dataset/onsets'

schluter_dataset_root_path = '/homedtic/rgong/cnnSyllableSeg/schulter'

schluter_audio_path = join(schluter_dataset_root_path, 'audio')
schluter_cv_path = join(schluter_dataset_root_path, 'splits')
schluter_annotations_path = join(schluter_dataset_root_path, 'annotations')

schluter_feature_data_path = join(schluter_dataset_root_path, 'feature')
