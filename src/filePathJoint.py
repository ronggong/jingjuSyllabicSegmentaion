from os.path import join, dirname
# from parameters import *

primarySchool_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/primary_school_recording'

primarySchool_wav_path = join(primarySchool_dataset_root_path, 'wav')
primarySchool_textgrid_path = join(primarySchool_dataset_root_path, 'textgrid')

root_path       = join(dirname(__file__),'..')

joint_cnn_model_path = join(root_path, 'cnnModels', 'joint')

scaler_joint_model_path = join(joint_cnn_model_path,
                                    'scaler_joint.pkl')

cnnModel_name = 'jan_joint'

eval_results_path = join(root_path, 'eval', 'results', cnnModel_name)

primarySchool_results_path = join(root_path, 'eval', 'joint', 'results')

full_path_keras_cnn_0 = join(joint_cnn_model_path, cnnModel_name)
