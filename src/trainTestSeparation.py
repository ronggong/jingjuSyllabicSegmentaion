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
from operator import itemgetter
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
    find the test recording numbers which meets the proportion
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


def getBoundaryNumber(textgrid_path, score_path):
    """
    output a list to show the syllable number for each aria,
    the syllable number is extracted from the textgrid
    the textgrid needs to have a score
    :param textgrid_path:
    :param score_path:
    :return:
    """
    listOnset = []
    list_file_path_name = []
    for file_path_name in os.walk(textgrid_path):
        list_file_path_name.append(file_path_name)

    list_artist_level_path = list_file_path_name[0][1]

    for artist_path in list_artist_level_path:

        textgrid_artist_path = join(textgrid_path, artist_path)
        recording_names = [f for f in os.listdir(textgrid_artist_path) if os.path.isfile(join(textgrid_artist_path, f))]

        for rn in recording_names:
            rn = rn.split('.')[0]
            groundtruth_textgrid_file = join(textgrid_path, artist_path, rn+'.TextGrid')
            if artist_path=='danAll' or artist_path=='laosheng':
                score_file = join(score_path, rn+'.csv')
            else:
                score_file = join(score_path, artist_path, rn + '.csv')

            if not os.path.isfile(score_file):
                continue
            # print(groundtruth_textgrid_file)

            lineList = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
            utteranceList = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

            # parse lines of groundtruth
            nestedUtteranceLists, numLines, numUtterances = textgridParser.wordListsParseByLines(lineList,
                                                                                                 utteranceList)

            # parse score
            utterance_durations, bpm = scoreParser.csvDurationScoreParser(score_file)

            # create the ground truth lab files
            numOnset = 0
            for idx, list in enumerate(nestedUtteranceLists):
                try:
                    if float(bpm[idx]):
                        print('Counting onset number ... ' + rn + ' phrase ' + str(idx + 1))

                        ul = list[1]
                        numOnsetLine = len(ul) - 1  # we don't count the first onset
                        numOnset += numOnsetLine
                except IndexError:
                    print(idx, 'not exist for recording', rn)

            listOnset += [[artist_path, rn, numOnset]]

    return listOnset


def getTestTrainRecordings():
    list_onset_nacta2017 = getBoundaryNumber(textgrid_path=nacta2017_textgrid_path, score_path=nacta2017_score_path)
    list_onset_nacta = getBoundaryNumber(textgrid_path=nacta_textgrid_path, score_path=nacta_score_path)

    listOnsetNacta2017Male = [n for n in list_onset_nacta2017 if '2017' in n[0] and 'ls' == n[1][:2]]
    listOnsetNacta2017Fem = [n for n in list_onset_nacta2017 if '2017' in n[0] and 'da' == n[1][:2]]
    numOnsetNacta2017Male = [n[2] for n in list_onset_nacta2017 if '2017' in n[0] and 'ls'==n[1][:2]]
    numOnsetNacta2017Fem = [n[2] for n in list_onset_nacta2017 if '2017' in n[0] and 'da'==n[1][:2]]

    listOnsetNactaMale = [n for n in list_onset_nacta if '2017' not in n[0] and 'ls' == n[1][:2]]
    listOnsetNactaFem = [n for n in list_onset_nacta if '2017' not in n[0] and 'da' == n[1][:2]]
    numOnsetNactaMale = [n[2] for n in list_onset_nacta if '2017' not in n[0] and 'ls'==n[1][:2]]
    numOnsetNactaFem = [n[2] for n in list_onset_nacta if '2017' not in n[0] and 'da'==n[1][:2]]

    print(len(numOnsetNacta2017Male), len(numOnsetNacta2017Fem), len(numOnsetNactaMale), len(numOnsetNactaFem))

    # segment the onset number list to accelerate the combination calculation
    numOnsetNacta2017Male0 = numOnsetNacta2017Male[:10]
    numOnsetNacta2017Male1 = numOnsetNacta2017Male[10:20]
    numOnsetNacta2017Male2 = numOnsetNacta2017Male[20:30]
    numOnsetNacta2017Male3 = numOnsetNacta2017Male[30:]

    numOnsetNacta2017Fem0 = numOnsetNacta2017Fem[:10]
    numOnsetNacta2017Fem1 = numOnsetNacta2017Fem[10:]

    numOnsetNactaMale0 = numOnsetNactaMale[:10]
    numOnsetNactaMale1 = numOnsetNactaMale[10:]

    numOnsetNactaFem0 = numOnsetNactaFem[:10]
    numOnsetNactaFem1 = numOnsetNactaFem[10:]

    # obtain the indices of test recordings
    print('test 0 nacta 2017 male number', testRecordings(numOnsetNacta2017Male0, 0.2)) # (1,7)
    print('test 1 nacta 2017 male number', testRecordings(numOnsetNacta2017Male1, 0.2)) # (11,16)
    print('test 2 nacta 2017 male number', testRecordings(numOnsetNacta2017Male2, 0.2)) # (23,28)
    print('test 3 nacta 2017 male number', testRecordings(numOnsetNacta2017Male3, 0.2)) # (31,34,35,39)
    print(sum(itemgetter(*[1,7,11,16,23,28,31,34,35,39])(numOnsetNacta2017Male))/float(sum(numOnsetNacta2017Male)))

    print('test 0 nacta 2017 female number', testRecordings(numOnsetNacta2017Fem0, 0.2)) # (0,1,2,7)
    print('test 1 nacta 2017 female number', testRecordings(numOnsetNacta2017Fem1, 0.2)) # (13,14)
    print(sum(itemgetter(*[0,1,2,7,13,14])(numOnsetNacta2017Fem))/float(sum(numOnsetNacta2017Fem)))

    print('test 0 nacta male number', testRecordings(numOnsetNactaMale0, 0.2)) # (0,8)
    print('test 1 nacta male number', testRecordings(numOnsetNactaMale1, 0.2)) # (16)
    print(sum(itemgetter(*[0,8,16])(numOnsetNactaMale))/float(sum(numOnsetNactaMale)))

    print('test 0 nacta female number', testRecordings(numOnsetNactaFem0, 0.2)) # (1,9)
    print('test 1 nacta female number', testRecordings(numOnsetNactaFem1, 0.2)) # (10)
    print(sum(itemgetter(*[1,9,10])(numOnsetNactaFem))/float(sum(numOnsetNactaFem)))

    recordingsTestNacta2017Male = [[listOnsetNacta2017Male[ii][0], listOnsetNacta2017Male[ii][1]] for ii in (1,7,11,16,23,28,31,34,35,39)]
    recordingsTrainNacta2017Male = [[listOnsetNacta2017Male[ii][0], listOnsetNacta2017Male[ii][1]] for ii in range(len(listOnsetNacta2017Male)) if ii not in (1,7,11,16,23,28,31,34,35,39)]
    # numTestQmMale = [lineOnsetQmMale[ii][1] for ii in (0, 1, 7, 8)]

    recordingsTestNacta2017Fem = [[listOnsetNacta2017Fem[ii][0], listOnsetNacta2017Fem[ii][1]] for ii in
                                   (0,1,2,7,13,14)]
    recordingsTrainNacta2017Fem = [[listOnsetNacta2017Fem[ii][0], listOnsetNacta2017Fem[ii][1]] for ii in
                                    range(len(listOnsetNacta2017Fem)) if ii not in (0,1,2,7,13,14)]
    # numTestQmFem = [lineOnsetQmFem[ii][1] for ii in (4, 8)]

    recordingsTestNactaMale = [[listOnsetNactaMale[ii][0], listOnsetNactaMale[ii][1]] for ii in
                                  (0,8,16)]
    recordingsTrainNactaMale = [[listOnsetNactaMale[ii][0], listOnsetNactaMale[ii][1]] for ii in
                                    range(len(listOnsetNactaMale)) if ii not in (0,8,16)]
    # numTestLon = [lineOnsetLon[ii][1] for ii in (2,)]

    recordingsTestNactaFem = [[listOnsetNactaFem[ii][0], listOnsetNactaFem[ii][1]] for ii in
                               (1,9,10)]
    recordingsTrainNactaFem = [[listOnsetNactaFem[ii][0], listOnsetNactaFem[ii][1]] for ii in
                                range(len(listOnsetNactaFem)) if ii not in (1,9,10)]
    # numTestBcn = [lineOnsetBcn[ii][1] for ii in (4, 5)]

    return recordingsTestNacta2017Male+recordingsTestNacta2017Fem, recordingsTestNactaMale+recordingsTestNactaFem, \
           recordingsTrainNacta2017Male+recordingsTrainNacta2017Fem, recordingsTrainNactaMale+recordingsTrainNactaFem


if __name__ == '__main__':
    # getTestTrainRecordings()
    # print getRecordingNames('TRAIN')

    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordings()

    print(testNacta2017)
    print(testNacta)
    print(trainNacta2017)
    print(trainNacta)