import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
from models import train_model_validation

if __name__ == '__main__':

    nlen = 15
    input_dim = (80, nlen)

    filename_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/feature_all_joint_subset.h5'
    filename_labels_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_joint_syllable_subset.pickle.gz'
    filename_sample_weights = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_joint_syllable_subset.pickle.gz'

    # filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all_joint_subset.h5'
    # filename_labels_train_validation_set = '/Users/gong/Documents/pycharmProjects/jingjuPhoneticSegmentation/training_data/labels_joint_syllable_subset.pickle.gz'
    # filename_sample_weights = '/Users/gong/Documents/pycharmProjects/jingjuPhoneticSegmentation/training_data/sample_weights_joint_syllable_subset.pickle.gz'

    tmp_train_validation_set = '/tmp/syllableSeg'
    os.mkdir(tmp_train_validation_set)
    filename_temp_train_validation_set = os.path.join(tmp_train_validation_set, 'feature_all_joint_subset.h5')

    shutil.copy2(filename_train_validation_set, filename_temp_train_validation_set)


    for running_time in range(3,5):
        # train the final model
        file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/jan_joint_syllable_subset'+str(running_time)+'.h5'
        file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/jan_joint_syllable_subset'+str(running_time)+'.csv'

        train_model_validation(filename_temp_train_validation_set,
                               filename_labels_train_validation_set,
                               filename_sample_weights,
                               filter_density=1,
                               dropout=0.5,
                               input_shape=input_dim,
                               file_path_model = file_path_model,
                               filename_log = file_path_log,
                               model_name='jan_original',
                               channel=1,
                               deep=True)

    shutil.rmtree(tmp_train_validation_set)