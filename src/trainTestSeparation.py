'''
 * Copyright (C) 2016  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuPhoneticSegmentationHMM
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''

import os
from itertools import combinations
from filePath import *
import numpy as np
import textgridParser
import scoreParser


def getRecordings(wav_path):
    recordings      = []
    for root, subFolders, files in os.walk(wav_path):
            for f in files:
                file_prefix, file_extension = os.path.splitext(f)
                if file_prefix != '.DS_Store' and file_prefix != '_DS_Store':
                    recordings.append(file_prefix)

    return recordings

def testRecordings(boundaries,proportion_testset):
    '''
    :param boundaries: a list of boundary number of each recording
    :param proportion_testset:
    :return: a list of test recordings
    '''

    sum_boundaries = sum(boundaries)
    boundaries     = np.array(boundaries)
    subsets        = []

    for ii in range(1,len(boundaries)):
        # print(ii, len(boundaries))
        for subset in combinations(range(len(boundaries)),ii):
            subsets.append([subset,abs(sum(boundaries[np.array(subset)])/float(sum_boundaries)-proportion_testset)])

    subsets        = np.array(subsets)
    subsets_sorted = subsets[np.argsort(subsets[:,1]),0]

    return subsets_sorted[0]


def getBoundaryNumber(recordings, dataset_path):

    listOnset = []
    for i_recording, recording_name in enumerate(recordings):
        groundtruth_textgrid_file   = os.path.join(aCapella_root, dataset_path, annotation_path, recording_name+'.TextGrid')
        score_file                  = os.path.join(aCapella_root, dataset_path, score_path,      recording_name+'.csv')

        if not os.path.isfile(score_file):
            print 'Score not found: ' + score_file
            continue

        lineList = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
        utteranceList = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

        # parse lines of groundtruth
        nestedUtteranceLists, numLines, numUtterances = textgridParser.wordListsParseByLines(lineList, utteranceList)

        # parse score
        utterance_durations, bpm = scoreParser.csvDurationScoreParser(score_file)

        # create the ground truth lab files
        numOnset = 0
        for idx, list in enumerate(nestedUtteranceLists):
            if int(bpm[idx]):
                print 'Counting onset number ... ' + recording_name + ' phrase ' + str(idx + 1)

                ul = list[1]
                numOnsetLine = len(ul)-1 # we don't count the first onset
                numOnset += numOnsetLine

        listOnset += [[recording_name, numOnset]]
    return listOnset

def getTestTrainRecordingsSex():
    """
    partition by male and female
    :return:
    """
    lineOnsetQmMale = getBoundaryNumber(queenMaryMale_Recordings, queenMarydataset_path)
    lineOnsetQmFem = getBoundaryNumber(queenMaryFem_Recordings, queenMarydataset_path)
    lineOnsetLon = getBoundaryNumber(london_Recordings, londonRecording_path)
    lineOnsetBcn = getBoundaryNumber(bcn_Recordings, bcnRecording_path)

    numOnsetQmMale = [n[1] for n in lineOnsetQmMale]
    numOnsetQmFem = [n[1] for n in lineOnsetQmFem]
    numOnsetLon = [n[1] for n in lineOnsetLon]
    numOnsetBcn = [n[1] for n in lineOnsetBcn]

    # print testRecordings(numOnsetQmMale, 0.2) # (0,1,7,8)
    # print testRecordings(numOnsetQmFem, 0.2) # (4,8)
    # print testRecordings(numOnsetLon, 0.2) # (2)
    # print testRecordings(numOnsetBcn, 0.2) # (4,5)

    recordingsTestQmMale = [lineOnsetQmMale[ii][0] for ii in (0, 1, 7, 8)]
    recordingsTrainQmMale = [lineOnsetQmMale[ii][0] for ii in range(len(lineOnsetQmMale)) if ii not in (0, 1, 7, 8)]
    numTestQmMale = [lineOnsetQmMale[ii][1] for ii in (0, 1, 7, 8)]

    recordingsTestQmFem = [lineOnsetQmFem[ii][0] for ii in (4, 8)]
    recordingsTrainQmFem = [lineOnsetQmFem[ii][0] for ii in range(len(lineOnsetQmFem)) if ii not in (4, 8)]
    numTestQmFem = [lineOnsetQmFem[ii][1] for ii in (4, 8)]

    recordingsTestLon = [lineOnsetLon[ii][0] for ii in (2,)]
    recordingsTrainLon = [lineOnsetLon[ii][0] for ii in range(len(lineOnsetLon)) if ii not in (2,)]
    numTestLon = [lineOnsetLon[ii][1] for ii in (2,)]

    recordingsTestBcn = [lineOnsetBcn[ii][0] for ii in (4,5)]
    recordingsTrainBcn = [lineOnsetBcn[ii][0] for ii in range(len(lineOnsetBcn)) if ii not in (4, 5)]
    numTestBcn = [lineOnsetBcn[ii][1] for ii in (4, 5)]

    print(recordingsTestQmMale)
    print(recordingsTrainQmMale)

    print(recordingsTestQmFem)
    print(recordingsTrainQmFem)

    print(recordingsTestBcn)
    print(recordingsTrainBcn)

    print(recordingsTestLon)
    print(recordingsTrainLon)

    numOnsetSum = sum(numOnsetQmMale) + sum(numOnsetQmFem) + sum(numOnsetLon) + sum(numOnsetBcn)
    numOnsetSumTest = sum(numTestQmMale) + sum(numTestQmFem) + sum(numTestLon) + sum(numTestBcn)

    print numOnsetSum
    print numOnsetSumTest

def recordingsNumTestTrainHelper(lineOnset, idx_test):
    recordingsTest = [lineOnset[ii][0] for ii in idx_test]
    recordingsTrain = [lineOnset[ii][0] for ii in range(len(lineOnset)) if
                              ii not in idx_test]
    numTest= [lineOnset[ii][1] for ii in idx_test]
    numTrain = [lineOnset[ii][1] for ii in range(len(lineOnset)) if ii not in idx_test]
    return recordingsTest, recordingsTrain, numTest, numTrain

def recordingNameMapping(recordings, dictFem, dictMale):
    nacta_out = []
    for r in recordings:
        if r in dictFem.keys():
            nacta_out.append(['danAll', dictFem[r]])
        if r in dictMale.keys():
            nacta_out.append(['laosheng', dictMale[r]])

    return nacta_out

def getTestTrainRecordingArtist():


    lineOnsetQmMale01 = getBoundaryNumber(queenMaryMale_01_Recordings, queenMarydataset_path)
    lineOnsetQmMale02 = getBoundaryNumber(queenMaryMale_02_Recordings, queenMarydataset_path)
    lineOnsetQmMale12 = getBoundaryNumber(queenMaryMale_12_Recordings, queenMarydataset_path)
    lineOnsetQmMale13 = getBoundaryNumber(queenMaryMale_13_Recordings, queenMarydataset_path)

    lineOnsetQmFem01 = getBoundaryNumber(queenMaryFem_01_Recordings, queenMarydataset_path)
    lineOnsetQmFem10 = getBoundaryNumber(queenMaryFem_10_Recordings, queenMarydataset_path)
    lineOnsetQmFem11 = getBoundaryNumber(queenMaryFem_11_Recordings, queenMarydataset_path)

    lineOnsetLonDan = getBoundaryNumber(londonDan_Recordings, londonRecording_path)
    lineOnsetLonLaosheng = getBoundaryNumber(londonLaosheng_Recordings, londonRecording_path)

    lineOnsetBcnDan = getBoundaryNumber(bcnDan_Recordings, bcnRecording_path)
    lineOnsetBcnLaosheng = getBoundaryNumber(bcnLaosheng_Recordings, bcnRecording_path)

    numOnsetQmMale01 = [n[1] for n in lineOnsetQmMale01]
    numOnsetQmMale02 = [n[1] for n in lineOnsetQmMale02]
    numOnsetQmMale12 = [n[1] for n in lineOnsetQmMale12]
    numOnsetQmMale13 = [n[1] for n in lineOnsetQmMale13]

    numOnsetQmFem01 = [n[1] for n in lineOnsetQmFem01]
    numOnsetQmFem10 = [n[1] for n in lineOnsetQmFem10]
    numOnsetQmFem11 = [n[1] for n in lineOnsetQmFem11]

    numOnsetLonDan = [n[1] for n in lineOnsetLonDan]
    numOnsetLonLaosheng = [n[1] for n in lineOnsetLonLaosheng]

    numOnsetBcnDan = [n[1] for n in lineOnsetBcnDan]
    numOnsetBcnLaosheng = [n[1] for n in lineOnsetBcnLaosheng]

    idx_test_QmMale01 = testRecordings(numOnsetQmMale01, 0.2)
    idx_test_QmMale13 = testRecordings(numOnsetQmMale13, 0.2)

    idx_test_QmFem01 = testRecordings(numOnsetQmFem01, 0.2)
    idx_test_QmFem10 = testRecordings(numOnsetQmFem10, 0.2)

    idx_test_LonDan = testRecordings(numOnsetLonDan, 0.2)
    idx_test_LonLaosheng = testRecordings(numOnsetLonLaosheng, 0.2)

    idx_test_BcnDan = testRecordings(numOnsetBcnDan, 0.2)
    idx_test_BcnLaosheng = testRecordings(numOnsetBcnLaosheng, 0.2)

    print(numOnsetQmMale02)
    print(numOnsetQmMale12)
    print(numOnsetQmFem11)

    testQmFem01, trainQmFem01, numTestQmFem01, numTrainQmFem01 = recordingsNumTestTrainHelper(lineOnsetQmFem01, idx_test_QmFem01)
    testQmFem10, trainQmFem10, numTestQmFem10, numTrainQmFem10 = recordingsNumTestTrainHelper(lineOnsetQmFem10, idx_test_QmFem10)

    testQmMale01, trainQmMale01, numTestQmMale01, numTrainQmMale01 = recordingsNumTestTrainHelper(lineOnsetQmMale01, idx_test_QmMale01)
    testQmMale13, trainQmMale13, numTestQmMale13, numTrainQmMale13 = recordingsNumTestTrainHelper(lineOnsetQmMale13, idx_test_QmMale13)

    testBcnDan, trainBcnDan, numTestBcnDan, numTrainBcnDan = recordingsNumTestTrainHelper(lineOnsetBcnDan, idx_test_BcnDan)
    testBcnLaosheng, trainBcnLaosheng, numTestBcnLaosheng, numTrainBcnLaosheng = recordingsNumTestTrainHelper(lineOnsetBcnLaosheng, idx_test_BcnLaosheng)

    testLonDan, trainLonDan, numTestLonDan, numTrainLonDan = recordingsNumTestTrainHelper(lineOnsetLonDan, idx_test_LonDan)
    testLonLaosheng, trainLonLaosheng, numTestLonLaosheng, numTrainLonLaosheng = recordingsNumTestTrainHelper(lineOnsetLonLaosheng, idx_test_LonLaosheng)

    testNacta = testQmFem01+testQmFem10+testQmMale01+testQmMale13+testLonDan+testLonLaosheng+testBcnDan+testBcnLaosheng
    trainNacta = trainQmFem01+trainQmFem10+trainQmMale01+trainQmMale13+trainLonDan+trainLonLaosheng+trainBcnDan+trainBcnLaosheng

    # numTestNacta = sum(numTestQmFem01+numTestQmFem10+numTestQmMale01+numTestQmMale13+numTestLonDan+numTestLonLaosheng+numTestBcnDan+numTestBcnLaosheng)
    # numTrainNacta = sum(numTrainQmFem01+numTrainQmFem10+numTrainQmMale01+numTrainQmMale13+numTrainLonDan+numTrainLonLaosheng+numTrainBcnDan+numTrainBcnLaosheng)

    # print(testNacta+queenMaryMale_02_Recordings+queenMaryFem_11_Recordings, numTestNacta+numOnsetQmMale02[0]+numOnsetQmFem11[0])
    # print(trainNacta+queenMaryMale_12_Recordings, numTrainNacta+numOnsetQmMale12[0])

    testNacta = testNacta + queenMaryMale_02_Recordings + queenMaryFem_11_Recordings
    trainNacta = trainNacta + queenMaryMale_12_Recordings

    dictFem = dict(dict_name_mapping_dan_qm)
    dictFem.update(dict_name_mapping_dan_bcn)
    dictFem.update(dict_name_mapping_dan_london)

    dictMale = dict(dict_name_mapping_laosheng_bcn)
    dictMale.update(dict_name_mapping_laosheng_qm)
    dictMale.update(dict_name_mapping_laosheng_london)

    testNacta_out = recordingNameMapping(testNacta, dictFem, dictMale)
    trainNacta_out = recordingNameMapping(trainNacta, dictFem, dictMale)

    print(testNacta_out)
    print(trainNacta_out)

    testNacta_out = [['danAll', 'daxp-Zhe_cai_shi-Suo_lin_nang01-qm'], ['danAll', 'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm'],
     ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm'], ['laosheng', 'lseh-Wei_guo_jia-Hong_yang_dong02-qm'],
     ['laosheng', 'lseh-Wo_ben_shi-Qiong_lin_yan-qm'], ['laosheng', 'lseh-Yi_lun_ming-Wen_zhao_guan-qm'],
     ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm'], ['danAll', 'dagbz-Feng_xiao_xiao-Yang_men_nv_jiang-lon'],
     ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying01-lon'], ['danAll', 'daspd-Du_shou_kong-Wang_jiang_ting-upf'],
     ['laosheng', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf'], ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm'],
     ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm']]
    trainNacta_out = [['danAll', 'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm'], ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm'],
     ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang04-qm'], ['danAll', 'daxp-Chun_qiu_ting-Suo_lin_nang01-qm'],
     ['danAll', 'danbz-Bei_jiu_chan-Chun_gui_men01-qm'], ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai02-qm'],
     ['laosheng', 'lseh-Tan_Yang_jia-Hong_yang_dong-qm'], ['laosheng', 'lseh-Zi_na_ri-Hong_yang_dong-qm'],
     ['laosheng', 'lsxp-Xi_ri_you-Zhu_lian_zhai-qm'], ['laosheng', 'lsxp-Quan_qian_sui-Gan_lu_si-qm'],
     ['laosheng', 'lsxp-Shi_ye_shuo-Ding_jun_shan-qm'], ['laosheng', 'lsxp-Wo_ben_shi-Kong_cheng_ji-qm'],
     ['laosheng', 'lsxp-Wo_zheng_zai-Kong_cheng_ji04-qm'], ['laosheng', 'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm'],
     ['danAll', 'daxp-Guan_Shi_yin-Tian_nv_san_hua-lon'], ['danAll', 'daeh-Yang_Yu_huan-Tai_zhen_wai_zhuan-lon'],
     ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu01-lon'], ['laosheng', 'lseh-Wei_guo_jia-Hong_yang_dong01-lon'],
     ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai01-upf'],
     ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-1-upf'], ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-2-upf'],
     ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan01-upf'], ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian02-qm']]

    return testNacta_out, trainNacta_out

def getRecordingNames(stringTrainTest='TRAIN'):
    if stringTrainTest == 'TRAIN':
        return [['male_01/neg_3', 'male_01/neg_4', 'male_01/neg_5', 'male_01/pos_1',
                 'male_01/pos_3', 'male_01/pos_6', 'male_02/neg_1', 'male_12/neg_1',
                 'male_13/pos_1', 'male_13/pos_3'] +
                ['fem_01/neg_1', 'fem_01/neg_3', 'fem_01/neg_5', 'fem_01/pos_1', 'fem_01/pos_5', 'fem_01/pos_7',
                 'fem_10/pos_1', 'fem_11/pos_1'],

                ['001', '007', '003', '004'],

                ['Dan-01', 'Dan-02', 'Dan-04', 'Laosheng-01', 'Laosheng-02']]

    if stringTrainTest == 'TEST':
        return [['male_01/neg_1', 'male_01/neg_2', 'male_01/pos_4', 'male_01/pos_5']+
                ['fem_01/pos_3', 'fem_10/pos_3'],

                ['005', '008'],

                ['Dan-03']]

if __name__ == '__main__':
    # getTestTrainRecordings()
    # print getRecordingNames('TRAIN')
    getTestTrainRecordingArtist()