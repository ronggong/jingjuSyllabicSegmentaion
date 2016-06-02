# -*- coding: utf-8 -*-
import sys, os
import numpy as np

sys.path.append(os.path.realpath('./src/'))

import textgridParser

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

def lineWordCount(textgrid_file):
    '''
    :param textgrid_file: annotation file
    :return: numLines, numWords, numDians
    '''

    numLines, numWords, numDians = 0,0,0
    wordDurationList, dianDurationList = [],[]

    entireLine      = textgridParser.textGrid2WordList(textgrid_file, whichTier='line')
    entireWordList  = textgridParser.textGrid2WordList(textgrid_file, whichTier='pinyin')
    entireDianList  = textgridParser.textGrid2WordList(textgrid_file, whichTier='dian')

    # parser word list for each line,
    if len(entireWordList):
        nestedWordLists, numLines, numWords = textgridParser.wordListsParseByLines(entireLine, entireWordList)
        wordDurationList                    = wordDuration(nestedWordLists)

    if len(entireDianList):
        nestedWordLists, numLines, numDians = textgridParser.wordListsParseByLines(entireLine, entireDianList)
        dianDurationList                    = wordDuration(nestedWordLists)

    return numLines, numWords, numDians, wordDurationList, dianDurationList

# ----------------------------------------------------------------------
# A Capella annotation data set
# 1. London recordings
# 2. Queen Mary Data set
# 3. Upf recordings
# 4. source separation /Users/gong/Documents/MTG document/Jingju arias/cleanSinging

londonSet       = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/londonRecording/annotation/'
upfSet          = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/bcnRecording/annotation'
queenmarySet    = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/QueenMary/jingjuSingingMono/annotation'

dataSets = [londonSet, upfSet, queenmarySet]

# ----------------------------------------------------------------------
# generate the file paths
filePaths       = []                                              # entire file paths
maleFilePaths   = []                                          # male singers file paths
femaleFilePaths = []                                        # female singers file paths

for ds in dataSets:
    for root, subFolders, files in os.walk(ds):
        for f in files:
            file_prefix, file_extension = os.path.splitext(f)
            if file_extension == '.TextGrid':
                filePath = os.path.join(root,f)
                filePaths.append(filePath)
                if 'Dan' in filePath or 'fem' in filePath or \
                                '001' in filePath or '007' in filePath:
                    femaleFilePaths.append(filePath)
                if 'Laosheng' in filePath or 'male' in filePath or \
                                '003' in filePath or '004' in filePath or \
                                '005' in filePath or '008' in filePath:
                    maleFilePaths.append(filePath)

# ----------------------------------------------------------------------
# total number of lines, words, dian for male and female singers
# mean, std of duration
nlSumMale, nwSumMale, ndSumMale         = 0,0,0
nlSumFemale, nwSumFemale, ndSumFemale   = 0,0,0
wdlMale, ddlMale                        = [],[]
wdlFemale, ddlFemale                    = [],[]

for tgfile in maleFilePaths:
    nl, nw, nd, wdl, ddl    = lineWordCount(tgfile)
    nlSumMale               += nl
    nwSumMale               += nw
    ndSumMale               += nd
    wdlMale                 += wdl
    ddlMale                 += ddl

for tgfile in femaleFilePaths:
    nl, nw, nd, wdl, ddl    = lineWordCount(tgfile)
    nlSumFemale             += nl
    nwSumFemale             += nw
    ndSumFemale             += nd
    wdlFemale               += wdl
    ddlFemale               += ddl

wdlTotal = wdlMale + wdlFemale
ddlTotal = ddlMale + ddlFemale

print 'Male total number of lines: {0}, words {1}, dian {2}'.format(nlSumMale,nwSumMale,ndSumMale)
print 'Female total number of lines: {0}, words {1}, dian {2}'.format(nlSumFemale,nwSumFemale,ndSumFemale)
print 'Total number of lines: {0}, words {1}, dian {2}'.format(nlSumFemale+nlSumMale,
                                                               nwSumFemale+nwSumMale,ndSumFemale+ndSumMale)
print 'Male average syllable duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(ddlMale),np.std(ddlMale),
                                                                            np.min(ddlMale),np.max(ddlMale))
print 'Female average syllable duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(ddlFemale),np.std(ddlFemale),
                                                                               np.min(ddlFemale),np.max(ddlFemale))
print 'Total average syllable duration: {0}, std {1}, min {2}, max {3}'.format(np.mean(ddlTotal),np.std(ddlTotal),
                                                                             np.min(ddlTotal),np.max(ddlTotal))