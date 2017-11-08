# -*- coding: utf-8 -*-
import sys, os
import numpy as np
from filePath import *

sys.path.append(os.path.realpath('./src/'))

import textgridParser
import scoreParser
from src.phonemeMap import nonvoicedconsonants
from src.trainTestSeparation import getTestTrainRecordingsNacta2017Artist

def wordDuration(nestedWordLists):
    '''
    :param nestedWordLists:     [[line0, wordList0], [line1, wordList1], ...]
    :return:                    list of word durations
    '''
    wordDurationList = []
    for nwl in nestedWordLists:
        for word in nwl[1]:
            duration = word[1] - word[0]
            wordDurationList.append(duration)

    return wordDurationList

def phoDuration(nestedWordLists):
    nvcDurationList = []
    vcDurationList  = []
    for nwl in nestedWordLists:
        for pho in nwl[1]:
            if pho[2] in nonvoicedconsonants:
                nvcDurationList.append(pho[1] - pho[0])
            else:
                vcDurationList.append(pho[1] - pho[0])
    return nvcDurationList, vcDurationList


def lineWordCount(textgrid_file):
    '''
    :param textgrid_file: annotation file
    :return: numLines, numWords, numDians
    '''

    numLines, numDians = 0,0
    dianDurationList = []

    entireLine      = textgridParser.textGrid2WordList(textgrid_file, whichTier='line')
    # entireDianList  = textgridParser.textGrid2WordList(textgrid_file, whichTier='dianSilence')
    entireWordList  = textgridParser.textGrid2WordList(textgrid_file, whichTier='pinyin')
    # entireDianList  = textgridParser.textGrid2WordList(textgrid_file, whichTier='dian')
    entireDianList  = textgridParser.textGrid2WordList(textgrid_file, whichTier='dianSilence')
    entirePhoList   = textgridParser.textGrid2WordList(textgrid_file, whichTier='details')


    if len(entireDianList):
        nestedWordLists, numLines, _ = textgridParser.wordListsParseByLines(entireLine, entireDianList)
        nestedWordLists_filtered = []
        numDians = 0
        for ii, wordList in enumerate(nestedWordLists):
            nestedWordLists_filtered.append(wordList)
            numDians += len(wordList[1])
        dianDurationList                    = wordDuration(nestedWordLists_filtered)


    return numLines, numDians, dianDurationList


# ----------------------------------------------------------------------
# generate the file paths
filePaths       = []                                              # entire file paths
maleFilePaths   = []                                          # male singers file paths
femaleFilePaths = []                                        # female singers file paths

testNacta2017, trainNacta2017 = getTestTrainRecordingsNacta2017Artist()

overallNacta2017 = trainNacta2017 + testNacta2017

laosheng_artists = ['20170327LiaoJiaNi', '20170418TianHao', '20170519LongTianMing', '20170519XuJingWei']

dan_artists = ['20170408SongRuoXuan', '20170418TianHao', '20170424SunYuZhu', '20170425SunYuZhu', '20170506LiuHaiLin', ]


# for artist_path in dan_artists:
#     textgrid_artist_path = join(nacta2017_textgrid_path, artist_path)
#     recording_names = [f for f in os.listdir(textgrid_artist_path) if os.path.isfile(join(textgrid_artist_path, f))]
#     for rn in recording_names:
#         rn = rn.split('.')[0]
#         textgrid_file = join(nacta2017_textgrid_path, artist_path, rn + '.TextGrid')
#         femaleFilePaths.append(textgrid_file)
#
# for artist_path in laosheng_artists:
#     textgrid_artist_path = join(nacta2017_textgrid_path, artist_path)
#     recording_names = [f for f in os.listdir(textgrid_artist_path) if
#                        os.path.isfile(join(textgrid_artist_path, f))]
#     for rn in recording_names:
#         rn = rn.split('.')[0]
#         textgrid_file = join(nacta2017_textgrid_path, artist_path, rn + '.TextGrid')
#         maleFilePaths.append(textgrid_file)

for recording in overallNacta2017:
    if recording[0] in dan_artists:
        textgrid_file = join(nacta2017_textgrid_path, recording[0], recording[1] + '.Textgrid')
        femaleFilePaths.append(textgrid_file)
    elif recording[0] in laosheng_artists:
        textgrid_file = join(nacta2017_textgrid_path, recording[0], recording[1] + '.Textgrid')
        maleFilePaths.append(textgrid_file)

# ----------------------------------------------------------------------
# total number of lines, words, dian for male and female singers
# mean, std of duration
nlSumMale, ndSumMale         = 0,0
nlSumFemale, ndSumFemale    = 0,0
ddlMale                     = []
ddlFemale                  = []

for ii, tgfile in enumerate(maleFilePaths):

    print(tgfile)
    nl, nd, ddl    = lineWordCount(tgfile)
    nlSumMale               += nl
    ndSumMale               += nd
    ddlMale                 += ddl


for ii, tgfile in enumerate(femaleFilePaths):

    nl, nd, ddl   = lineWordCount(tgfile)
    nlSumFemale             += nl
    ndSumFemale             += nd
    ddlFemale               += ddl


ddlTotal = ddlMale + ddlFemale


print "number of recordings: {0}".format(len(overallNacta2017))
print 'Male total number of lines: {0}, dian {1}'.format(nlSumMale, ndSumMale)
print 'Female total number of lines: {0}, dian {1}'.format(nlSumFemale,ndSumFemale)
print 'Total number of lines: {0}, dian {1}'.format(nlSumFemale+nlSumMale, ndSumFemale+ndSumMale)

print 'Male average syllable duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(ddlMale),np.std(ddlMale),
                                                                            np.min(ddlMale),np.max(ddlMale))
print 'Female average syllable duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(ddlFemale),np.std(ddlFemale),
                                                                               np.min(ddlFemale),np.max(ddlFemale))
print 'Total average syllable duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(ddlTotal),np.std(ddlTotal),
                                                                             np.min(ddlTotal),np.max(ddlTotal))

