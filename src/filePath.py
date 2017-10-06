from os.path import join

# audio and annotation root path

# dataset_root_path     = 'to/your/root/path'

# dataset_textgrid_path = dataset_root_path
# aCapella_root         = dataset_root_path

# text grid path which contains phoneme annotations
dataset_textgrid_path = '/Users/gong/Documents/MTG document/Jingju arias/aCappella-dataset-syllable-cnn/'

textgrid_path_dan           = join(dataset_textgrid_path,'textgrid','danAll')
textgrid_path_laosheng      = join(dataset_textgrid_path,'textgrid','qmLonUpf/laosheng')

# aCapella root
aCapella_root   = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/'

# dataset path
queenMarydataset_path    = 'QueenMary/jingjuSingingMono'
londonRecording_path     = 'londonRecording'
bcnRecording_path        = 'bcnRecording'

segPhrase_path          = 'segPhrase'
annotation_path         = 'annotation'
audio_path              = 'audio'
score_path              = 'scoreDianSilence'
groundtruth_lab_path    = 'groundtruth_lab'
eval_details_path       = 'eval_details'

# Audio recordings dataset path
# Queen Mary recordings
queenMaryFem_01_Recordings = ['fem_01/neg_1', 'fem_01/neg_3', 'fem_01/neg_5', 'fem_01/pos_1', 'fem_01/pos_3', 'fem_01/pos_5', 'fem_01/pos_7']
queenMaryFem_10_Recordings = ['fem_10/pos_1', 'fem_10/pos_3']
queenMaryFem_11_Recordings = ['fem_11/pos_1']

queenMaryMale_01_Recordings = ['male_01/neg_1','male_01/neg_2','male_01/neg_3','male_01/neg_4','male_01/neg_5',
                            'male_01/pos_1','male_01/pos_2','male_01/pos_3','male_01/pos_4','male_01/pos_5','male_01/pos_6']
queenMaryMale_02_Recordings = ['male_02/neg_1']
queenMaryMale_12_Recordings = ['male_12/neg_1']
queenMaryMale_13_Recordings = ['male_13/pos_1', 'male_13/pos_3']

queenMaryFem_Recordings     = queenMaryFem_01_Recordings + queenMaryFem_10_Recordings + queenMaryFem_11_Recordings
queenMaryMale_Recordings    = queenMaryMale_01_Recordings + queenMaryMale_02_Recordings + queenMaryMale_12_Recordings + queenMaryMale_13_Recordings

queenMary_Recordings        = queenMaryFem_Recordings + queenMaryMale_Recordings

queenMary_Recordings_train_male = ['male_01/neg_3', 'male_01/neg_4', 'male_01/neg_5', 'male_01/pos_1',
                                 'male_01/pos_3', 'male_01/pos_6', 'male_02/neg_1', 'male_12/neg_1',
                                 'male_13/pos_1', 'male_13/pos_3']
queenMary_Recordings_train_fem  = ['fem_01/neg_1', 'fem_01/neg_3', 'fem_01/neg_5', 'fem_01/pos_1',
                                   'fem_01/pos_5', 'fem_01/pos_7', 'fem_10/pos_1', 'fem_11/pos_1']

queenMary_Recordings_test_male          = ['male_01/neg_1', 'male_01/neg_2', 'male_01/pos_4', 'male_01/pos_5']
queenMary_Recordings_test_fem           = ['fem_01/pos_3', 'fem_10/pos_3']

queenMary_Recordings_test               = queenMary_Recordings_test_male + queenMary_Recordings_test_fem


# London recordings
londonDan_Recordings        = ['Dan-01', 'Dan-02', 'Dan-03', 'Dan-04']
londonLaosheng_Recordings   = ['Laosheng-01', 'Laosheng-02', 'Laosheng-04']

london_Recordings           = londonDan_Recordings + londonLaosheng_Recordings

london_Recordings_train_male = ['Laosheng-01', 'Laosheng-02', 'Laosheng-04']
london_Recordings_train_fem  = ['Dan-01', 'Dan-02', 'Dan-04']
london_Recordings_test_fem  = ['Dan-03']
london_Recordings_test      = london_Recordings_test_fem

# bcn recordings
bcnDan_Recordings           = ['001', '007']
bcnLaosheng_Recordings      = ['003', '004', '005', '008']

bcn_Recordings              = bcnDan_Recordings + bcnLaosheng_Recordings
bcn_Recordings_train_male   = ['003', '004']
bcn_Recordings_train_fem    = ['001', '007']
bcn_Recordings_test_male    = ['005', '008']
bcn_Recordings_test         = bcn_Recordings_test_male

dict_name_mapping_dan_qm    = {'fem_01/neg_1':'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm',
                               'fem_01/neg_3':'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm',
                               'fem_01/neg_5':'daxp-Jiao_Zhang_sheng-Hong_niang04-qm',
                               'fem_01/pos_1':'daxp-Chun_qiu_ting-Suo_lin_nang01-qm',
                               'fem_01/pos_3':'daxp-Zhe_cai_shi-Suo_lin_nang01-qm',
                               'fem_01/pos_5':'danbz-Bei_jiu_chan-Chun_gui_men01-qm',
                               'fem_01/pos_7':'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm',
                               'fem_10/pos_1':'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm',
                               'fem_10/pos_3':'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai02-qm',
                               'fem_11/pos_1':'daxp-Jiao_Zhang_sheng-Hong_niang01-qm'}
dict_name_mapping_laosheng_qm = {'male_01/neg_1':'lseh-Wei_guo_jia-Hong_yang_dong02-qm',
                                 'male_01/neg_2':'lseh-Tan_Yang_jia-Hong_yang_dong-qm',
                                 'male_01/neg_3':'lseh-Zi_na_ri-Hong_yang_dong-qm',
                                 'male_01/neg_4':'lseh-Wo_ben_shi-Qiong_lin_yan-qm',
                                 'male_01/neg_5':'lseh-Yi_lun_ming-Wen_zhao_guan-qm',
                                 'male_01/pos_1':'lsxp-Xi_ri_you-Zhu_lian_zhai-qm',
                                 'male_01/pos_2':'lsxp-Shen_gong_wu-Gan_lu_si-qm',
                                 'male_01/pos_3':'lsxp-Quan_qian_sui-Gan_lu_si-qm',
                                 'male_01/pos_4':'lsxp-Shi_ye_shuo-Ding_jun_shan-qm',
                                 'male_01/pos_5':'lsxp-Wo_ben_shi-Kong_cheng_ji-qm',
                                 'male_01/pos_6':'lsxp-Wo_zheng_zai-Kong_cheng_ji04-qm',
                                 'male_02/neg_1':'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm',
                                 'male_12/neg_1':'lsxp-Jiang_shen_er-San_jia_dian02-qm',
                                 'male_13/pos_1':'lsxp-Huai_nan_wang-Huai_he_ying02-qm',
                                 'male_13/pos_3':'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm'}
dict_name_mapping_dan_london = {'Dan-01':'daxp-Guan_Shi_yin-Tian_nv_san_hua-lon',
                                'Dan-02':'daeh-Yang_Yu_huan-Tai_zhen_wai_zhuan-lon',
                                'Dan-03':'dagbz-Feng_xiao_xiao-Yang_men_nv_jiang-lon',
                                'Dan-04':'daspd-Hai_dao_bing-Gui_fei_zui_jiu01-lon'}
dict_name_mapping_laosheng_london = {'Laosheng-01':'lseh-Wei_guo_jia-Hong_yang_dong01-lon',
                                     'Laosheng-02':'lsxp-Huai_nan_wang-Huai_he_ying01-lon',
                                     # 'Laosheng-03':'lsxp-Chao_xia_ying-Sha_jia_bang-lon',
                                     'Laosheng-04':'lsfxp-Yang_si_lang-Si_lang_tan_mu-lon'}
dict_name_mapping_dan_bcn = {'001':'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai01-upf',
                             '007':'daspd-Du_shou_kong-Wang_jiang_ting-upf'}
dict_name_mapping_laosheng_bcn = {'003':'lsxp-Jiang_shen_er-San_jia_dian01-1-upf',
                                  '004':'lsxp-Jiang_shen_er-San_jia_dian01-2-upf',
                                  '005':'lsxp-Guo_liao_yi-Wen_zhao_guan01-upf',
                                  '008':'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf'}

# current path
from os.path import join,dirname
from parameters import mth_ODF, fusion, layer2, filter_shape

root_path       = join(dirname(__file__),'..')
cnnModels_path  = join(root_path, 'cnnModels')

if mth_ODF == 'jan':
    filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jan_deep_class_weight_mfccBands_2D_all_old_ismir.h5'
    cnnModel_name         = 'jan_deep_old_ismir'
elif mth_ODF == 'jan_chan3':
    filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jan_class_weight_3_chans_mfccBands_2D_all_optim.h5'
    # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jan_class_weight_3_chans_layer1_70_mfccBands_2D_all_optim.h5'
    # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jan_no_class_weight_mfccBands_2D_all_optim.h5'
    cnnModel_name           = 'jan_cw_3_chans'
elif mth_ODF == 'jordi_horizontal_timbral':
    if layer2 == 20:
        filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_horizontal_timbral_filter_layer2_20_mfccBands_2D_all_optim.h5'
        cnnModel_name        = 'jordi_cw_conv_dense_horizontal_timbral_filter_layer2_20'
    else:
        filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_horizontal_timbral_filter_mfccBands_2D_all_optim.h5'
        cnnModel_name         = 'jordi_cw_conv_dense_horizontal_timbral_filter'
else:
    # mth_ODF == 'jordi'
    if fusion:
        if layer2 == 20:
            filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_layer2_20_mfccBands_2D_all_optim.h5'
            filename_keras_cnn_1 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_timbral_filter_layer2_20_mfccBands_2D_all_optim.h5'
            cnnModel_name        = 'jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_multiply_layer2_20'
        else:
            filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_mfccBands_2D_all_optim.h5'
            filename_keras_cnn_1 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_timbral_filter_mfccBands_node04_2D_all_optim.h5'
            cnnModel_name        = 'jordi_fusion_old_ismir'
    else:
        if filter_shape == 'temporal':
            # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_mfccBands_2D_all_optim.h5'
            # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_with_dense_mfccBands_2D_all_optim.h5'
            if layer2 == 20:
                filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_layer2_20_mfccBands_2D_all_optim.h5'
                cnnModel_name           = 'jordi_cw_conv_dense_layer2_20'
            else:
                # layer2 32 nodes
                filename_keras_cnn_0    = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_mfccBands_2D_all_optim.h5'
                # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_149k_mfccBands_2D_all_optim.h5'
                # filename_keras_cnn_1  = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_second_model_32_mfccBands_2D_all_optim.h5'
                cnnModel_name           = 'jordi_temporal_old_ismir'
        else:
            # timbral filter shape
            if layer2 == 20:
                filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_timbral_filter_layer2_20_mfccBands_2D_all_optim.h5'
                cnnModel_name           = 'jordi_cw_conv_dense_timbral_filter_layer2_20'
            else:
                # layer2 32 nodes
                filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_timbral_filter_mfccBands_node04_2D_all_optim.h5'
                # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_timbral_filter_152k_mfccBands_2D_all_optim.h5'
                # filename_keras_cnn_1 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_timbral_filter_second_model_32_mfccBands_2D_all_optim.h5'
                cnnModel_name          = 'jordi_timbral_old_ismir'

filename_scaler_onset    = 'scaler_syllable_mfccBands2D.pkl'
filename_scaler_onset_23 = 'scaler_syllable_mfccBands2D_23.pkl'
filename_scaler_onset_46 = 'scaler_syllable_mfccBands2D_46.pkl'
filename_scaler_onset_93 = 'scaler_syllable_mfccBands2D_93.pkl'


full_path_keras_cnn_0                   = join(cnnModels_path, filename_keras_cnn_0)

if fusion and mth_ODF == 'jordi':
    full_path_keras_cnn_1                   = join(cnnModels_path, filename_keras_cnn_1)

full_path_mfccBands_2D_scaler_onset     = join(cnnModels_path, filename_scaler_onset)

full_path_mfccBands_2D_scaler_onset_23     = join(cnnModels_path, filename_scaler_onset_23)
full_path_mfccBands_2D_scaler_onset_46     = join(cnnModels_path, filename_scaler_onset_46)
full_path_mfccBands_2D_scaler_onset_93     = join(cnnModels_path, filename_scaler_onset_93)


# Evaluation path
# cnnModel_name           = 'jan_cw'
# cnnModel_name           = 'jan_cw_3_chans'
# cnnModel_name           = 'jan_cw_3_chans_layer1_70'
# cnnModel_name           = 'jan_ncw'
# cnnModel_name           = 'jordi_cw'
# cnnModel_name           = 'jordi_ncw'
# cnnModel_name           = 'jordi_cw_dense'
# cnnModel_name           = 'jordi_cw_conv_dense'
# cnnModel_name           = 'jordi_cw_conv_dense_149k'

# cnnModel_name           = 'jordi_cw_conv_dense_layer2_20'

# cnnModel_name           = 'jordi_cw_conv_dense_timbral_filter'
# cnnModel_name           = 'jordi_cw_conv_dense_timbral_filter_152k'
# cnnModel_name           = 'jordi_cw_conv_dense_timbral_filter_layer2_20'
# cnnModel_name           = 'jordi_cw_conv_dense_timbral_filter_late_fusion_2_models_multiply'

# cnnModel_name           = 'jordi_cw_conv_dense_horizontal_timbral_filter'
# cnnModel_name           = 'jordi_cw_conv_dense_horizontal_timbral_filter_layer2_20'

# cnnModel_name           = 'jordi_cw_conv_dense_horizontal_filter_late_fusion_2_models_multiply'
# cnnModel_name           = 'jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_multiply_layer2_20'

decoding_method         = '_win'

eval_results_path       = join(root_path, 'eval', 'results', cnnModel_name+decoding_method)
eval_fig_path           = join(root_path, 'eval', 'figs', cnnModel_name+decoding_method)


# acoustic model name and path
filename_scaler_am                      = 'scaler_syllableSeg_am_mfccBands2D.pkl'
full_path_mfccBands_2D_scaler_am        = join(cnnModels_path, 'am', filename_scaler_am)
filename_keras_cnn_am                   = 'keras.cnn_syllableSeg_am_mfccBands_2D_all_optim.h5'
full_path_keras_cnn_am                  = join(cnnModels_path, 'am', filename_keras_cnn_am)
