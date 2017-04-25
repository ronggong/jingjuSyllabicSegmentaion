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

def getTestTrainRecordings():
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
    print getRecordingNames('TRAIN')