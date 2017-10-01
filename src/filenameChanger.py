"""
functions to change the filenames
"""

from filePath import *
import os
import shutil

def scoreFilenameChangerHelper(dict_name_mapping, score_full_path, score_new_path, role='danAll'):
    """
    copy file from full_fn_pinyin to new path, change name
    :param dict_name_mapping:
    :param score_full_path:
    :param score_new_path:
    :param role:
    :return:
    """
    for fn in dict_name_mapping:
        full_fn_pinyin = os.path.join(score_full_path, fn+'_pinyin.csv')
        try:
            shutil.copy2(full_fn_pinyin, os.path.join(score_new_path, role, dict_name_mapping[fn]+'.csv'))  # complete target filename given
        except IOError:
            print('filename not found', full_fn_pinyin)

def scoreFilenameChanger():
    qm_score_full_path = os.path.join(aCapella_root, queenMarydataset_path, score_path)
    lon_score_full_path = os.path.join(aCapella_root, londonRecording_path, score_path)
    bcn_score_full_path = os.path.join(aCapella_root, bcnRecording_path, score_path)

    score_new_path = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset/scoreDianSilence_pinyin'

    scoreFilenameChangerHelper(dict_name_mapping_dan_qm, qm_score_full_path, score_new_path, 'danAll')
    scoreFilenameChangerHelper(dict_name_mapping_dan_london, lon_score_full_path, score_new_path, 'danAll')
    scoreFilenameChangerHelper(dict_name_mapping_dan_bcn, bcn_score_full_path, score_new_path, 'danAll')

    scoreFilenameChangerHelper(dict_name_mapping_laosheng_qm, qm_score_full_path, score_new_path, 'laosheng')
    scoreFilenameChangerHelper(dict_name_mapping_laosheng_london, lon_score_full_path, score_new_path, 'laosheng')
    scoreFilenameChangerHelper(dict_name_mapping_laosheng_bcn, bcn_score_full_path, score_new_path, 'laosheng')

if __name__ == '__main__':
    scoreFilenameChanger()