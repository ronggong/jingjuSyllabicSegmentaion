import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import train_model_validation


if __name__ == '__main__':

    nlen = 15
    input_dim = (80, nlen)

    # filename_train_validation_set = '/scratch/rgongcnnSyllableSeg_jan_madmom_original_deep/syllableSeg/feature_all_artist_filter_madmom.h5'
    # filename_labels_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_train_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'
    # filename_sample_weights = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'

    # filename_train_validation_set = '/scratch/rgongcnnSyllableSeg_jan_madmom_original_deep/syllableSeg/feature_all_ismir_madmom.h5'
    # filename_labels_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_train_set_all_syllableSeg_mfccBands2D_old+new_ismir_madmom.pickle.gz'
    # filename_sample_weights = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_syllableSeg_mfccBands2D_old+new_ismir_madmom.pickle.gz'

    # file_path_model = '../../temp/keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_artist_filter_madmom.h5'
    # file_path_log = '../../temp/keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_artist_filter_madmom.csv'

    filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all_artist_filter_madmom.h5'
    filename_labels_train_validation_set = '../../trainingData/labels_train_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'
    filename_sample_weights = '../../trainingData/sample_weights_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'

    # filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all_ismir_madmom.h5'
    # filename_labels_train_validation_set = '../../trainingData/labels_train_set_all_syllableSeg_mfccBands2D_old+new_ismir_madmom.pickle.gz'
    # filename_sample_weights = '../../trainingData/sample_weights_syllableSeg_mfccBands2D_old+new_ismir_madmom.pickle.gz'

    for running_time in range(5):
        # train the final model
        file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/keras.cnn_syllableSeg_jan_artist_filter_deep_strong_front'+str(running_time)+'.h5'
        file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/keras.cnn_syllableSeg_jan_artist_filter_deep_strong_front'+str(running_time)+'.csv'

        # file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/keras.cnn_syllableSeg_jan_ismir_less_deep' + str(
        #     running_time) + '.h5'
        # file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/keras.cnn_syllableSeg_jan_ismir_less_deep'+str(running_time)+'.csv'

        train_model_validation(filename_train_validation_set,
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