"""
check the score sanity
"""

from src.filePath import *
from src.trainTestSeparation import getTestTrainRecordingsArtist
from src.textgridParser import textGrid2WordList, syllableTextgridExtraction
from src.scoreParser import csvScorePinyinParser

def scorePinyinChecker(score_pinyin_list, textgrid_pinyin_list):
    if score_pinyin_list == textgrid_pinyin_list:
        return True
    else:
        return False

def batchScorePinyinCheck():
    """
    check the score pinyin and textgrid pinyin coherence
    :return:
    """
    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtist()

    for artist_path, rn in trainNacta2017:
        score_file  = join(nacta2017_score_pinyin_path, artist_path, rn+'.csv')

        syllables, pinyins, syllable_durations, bpm = csvScorePinyinParser(score_file)
        nestedSyllableLists, numLines, numSyllables = syllableTextgridExtraction(join(nacta2017_textgrid_path, artist_path), rn, 'line', 'dianSilence')

        # print(pinyins)
        print(artist_path, rn)
        for ii, nsl in enumerate(nestedSyllableLists):
            if nsl[1]:
                textgrid_pinyin_list = [sl[2] for sl in nsl[1]]
                score_pinyin_list = [s for s in pinyins[ii] if len(s)]
                if not scorePinyinChecker(score_pinyin_list, textgrid_pinyin_list):
                    print(ii)
                    print(score_pinyin_list)
                    print(textgrid_pinyin_list)

if __name__ == '__main__':
    print(batchScorePinyinCheck())



