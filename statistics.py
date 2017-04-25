# -*- coding: utf-8 -*-
import sys, os
import numpy as np
from filePath import *

sys.path.append(os.path.realpath('./src/'))

import textgridParser
import scoreParser
from src.phonemeMap import nonvoicedconsonants

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


def lineWordCount(textgrid_file, score_file):
    '''
    :param textgrid_file: annotation file
    :return: numLines, numWords, numDians
    '''

    numLines, numWords, numDians, numPhos = 0,0,0,0
    wordDurationList, dianDurationList, nvcDurationList, vcDurationList = [],[],[],[]

    entireLine      = textgridParser.textGrid2WordList(textgrid_file, whichTier='line')
    entireWordList  = textgridParser.textGrid2WordList(textgrid_file, whichTier='pinyin')
    entireDianList  = textgridParser.textGrid2WordList(textgrid_file, whichTier='dian')
    entirePhoList   = textgridParser.textGrid2WordList(textgrid_file, whichTier='details')

    utterance_durations, bpm = scoreParser.csvDurationScoreParser(score_file)

    # parser word list for each line,
    if len(entireWordList):
        nestedWordLists, numLines, _ = textgridParser.wordListsParseByLines(entireLine, entireWordList)
        nestedWordLists_filtered = []
        numWords = 0
        for ii, wordList in enumerate(nestedWordLists):
            if int(bpm[ii]):
                # omit non score lines
                nestedWordLists_filtered.append(wordList)
                numWords += len(wordList[1])
        numLines = len(nestedWordLists_filtered)

        wordDurationList                    = wordDuration(nestedWordLists_filtered)

    if len(entireDianList):
        nestedWordLists, _, _ = textgridParser.wordListsParseByLines(entireLine, entireDianList)
        nestedWordLists_filtered = []
        numDians = 0
        for ii, wordList in enumerate(nestedWordLists):
            if int(bpm[ii]):
                nestedWordLists_filtered.append(wordList)
                numDians += len(wordList[1])
        dianDurationList                    = wordDuration(nestedWordLists_filtered)

    if len(entirePhoList):
        nestedWordLists, _, _ = textgridParser.wordListsParseByLines(entireLine, entirePhoList)
        nestedWordLists_filtered = []
        numPhos = 0
        for ii, wordList in enumerate(nestedWordLists):
            if int(bpm[ii]):
                nestedWordLists_filtered.append(wordList)
                numPhos += len(wordList[1])
        nvcDurationList, vcDurationList                    = phoDuration(nestedWordLists_filtered)

    return numLines, numWords, numDians, numPhos, wordDurationList, dianDurationList, nvcDurationList, vcDurationList

# ----------------------------------------------------------------------
# A Capella annotation data set
# 1. London recordings
# 2. Queen Mary Data set
# 3. Upf recordings
# 4. source separation /Users/gong/Documents/MTG document/Jingju arias/cleanSinging


# ----------------------------------------------------------------------
# generate the file paths
filePaths       = []                                              # entire file paths
maleFilePaths   = []                                          # male singers file paths
femaleFilePaths = []                                        # female singers file paths
maleScorePaths  = []
femaleScorePaths = []

# collect score and textgrid full paths
def collectTextgridScorePath(dict_name_mapping,
                             textgrid_path_collection,
                             score_path_collection,
                             textgrid_path,
                             dataset_path):
    inv_dict_name_mapping = {v: k for k, v in dict_name_mapping.iteritems()}
    for file_name in dict_name_mapping.values():
        textgrid_path_collection.append(join(textgrid_path, file_name + '.textgrid'))
        score_path_collection.append(
            join(aCapella_root, dataset_path, score_path, inv_dict_name_mapping[file_name] + '.csv'))
    return textgrid_path_collection, score_path_collection

femaleFilePaths, femaleScorePaths = collectTextgridScorePath(dict_name_mapping_dan_qm,
                                                           femaleFilePaths,
                                                           femaleScorePaths,
                                                             textgrid_path_dan,
                                                             queenMarydataset_path)

femaleFilePaths, femaleScorePaths = collectTextgridScorePath(dict_name_mapping_dan_london,
                                                           femaleFilePaths,
                                                           femaleScorePaths,
                                                             textgrid_path_dan,
                                                             londonRecording_path)

femaleFilePaths, femaleScorePaths = collectTextgridScorePath(dict_name_mapping_dan_bcn,
                                                           femaleFilePaths,
                                                           femaleScorePaths,
                                                             textgrid_path_dan,
                                                             bcnRecording_path)

maleFilePaths, maleScorePaths = collectTextgridScorePath(dict_name_mapping_laosheng_qm,
                                                           maleFilePaths,
                                                           maleScorePaths,
                                                             textgrid_path_laosheng,
                                                             queenMarydataset_path)

maleFilePaths, maleScorePaths = collectTextgridScorePath(dict_name_mapping_laosheng_london,
                                                           maleFilePaths,
                                                           maleScorePaths,
                                                             textgrid_path_laosheng,
                                                             londonRecording_path)

maleFilePaths, maleScorePaths = collectTextgridScorePath(dict_name_mapping_laosheng_bcn,
                                                           maleFilePaths,
                                                           maleScorePaths,
                                                             textgrid_path_laosheng,
                                                             bcnRecording_path)


# ----------------------------------------------------------------------
# total number of lines, words, dian for male and female singers
# mean, std of duration
nlSumMale, nwSumMale, ndSumMale, nphoSumMale         = 0,0,0,0
nlSumFemale, nwSumFemale, ndSumFemale, nphoSumFemale   = 0,0,0,0
wdlMale, ddlMale, nvcdlMale, vcdlMale                        = [],[],[],[]
wdlFemale, ddlFemale, nvcdlFemale, vcdlFemale                    = [],[],[],[]

for ii, tgfile in enumerate(maleFilePaths):
    score_file              = maleScorePaths[ii]
    if not os.path.isfile(score_file):
        print 'Score not found: ' + score_file
        continue
    print(tgfile)
    print(score_file)
    nl, nw, nd, npho, wdl, ddl, nvcdl, vcdl    = lineWordCount(tgfile, score_file)
    nlSumMale               += nl
    nwSumMale               += nw
    ndSumMale               += nd
    nphoSumMale             += npho
    wdlMale                 += wdl
    ddlMale                 += ddl
    nvcdlMale               += nvcdl
    vcdlMale                += vcdl

for ii, tgfile in enumerate(femaleFilePaths):
    score_file = femaleScorePaths[ii]
    if not os.path.isfile(score_file):
        print 'Score not found: ' + score_file
        continue

    nl, nw, nd, npho, wdl, ddl, nvcdl, vcdl    = lineWordCount(tgfile, score_file)
    nlSumFemale             += nl
    nwSumFemale             += nw
    ndSumFemale             += nd
    nphoSumFemale           += npho
    wdlFemale               += wdl
    ddlFemale               += ddl
    nvcdlFemale             += nvcdl
    vcdlFemale              += vcdl

wdlTotal = wdlMale + wdlFemale
ddlTotal = ddlMale + ddlFemale
nvcdlTotal  = nvcdlMale + nvcdlFemale
vcdlTotal   = vcdlMale + vcdlFemale

print 'Male total number of lines: {0}, words {1}, dian {2}, nvcpho {3}, vcpho {4}'.format(nlSumMale,nwSumMale,ndSumMale,len(nvcdlMale),len(vcdlMale))
print 'Female total number of lines: {0}, words {1}, dian {2}, nvcpho {3}, vcpho {4}'.format(nlSumFemale,nwSumFemale,ndSumFemale,len(nvcdlFemale),len(vcdlFemale))
print 'Total number of lines: {0}, words {1}, dian {2}, nvcpho {3}, vcpho {4}'.format(nlSumFemale+nlSumMale,
                                                               nwSumFemale+nwSumMale,
                                                                        ndSumFemale+ndSumMale,
                                                                        len(nvcdlMale)+len(nvcdlFemale),
                                                                        len(vcdlMale)+len(vcdlFemale))
print 'Male average syllable duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(ddlMale),np.std(ddlMale),
                                                                            np.min(ddlMale),np.max(ddlMale))
print 'Female average syllable duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(ddlFemale),np.std(ddlFemale),
                                                                               np.min(ddlFemale),np.max(ddlFemale))
print 'Total average syllable duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(ddlTotal),np.std(ddlTotal),
                                                                             np.min(ddlTotal),np.max(ddlTotal))

print 'Male average unvoiced pho duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(nvcdlMale),np.std(nvcdlMale),
                                                                            np.min(nvcdlMale),np.max(nvcdlMale))
print 'Female average unvoiced pho duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(nvcdlFemale),np.std(nvcdlFemale),
                                                                               np.min(nvcdlFemale),np.max(nvcdlFemale))
print 'Total average unvoiced pho duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(nvcdlTotal),np.std(nvcdlTotal),
                                                                             np.min(nvcdlTotal),np.max(nvcdlTotal))

print 'Male average voiced pho duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(vcdlMale),np.std(vcdlMale),
                                                                            np.min(vcdlMale),np.max(vcdlMale))
print 'Female average voiced pho duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(vcdlFemale),np.std(vcdlFemale),
                                                                               np.min(vcdlFemale),np.max(vcdlFemale))
print 'Total average voiced pho duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(vcdlTotal),np.std(vcdlTotal),
                                                                             np.min(vcdlTotal),np.max(vcdlTotal))