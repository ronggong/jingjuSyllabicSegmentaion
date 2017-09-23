import csv
import pinyin
import os

def csvDurationScoreParser(scoreFilename):

    syllable_durations = []
    bpm                 = []

    with open(scoreFilename, 'rb') as csvfile:
        score = csv.reader(csvfile)
        for idx, row in enumerate(score):
            if idx%2:
                syllable_durations.append(row[1:])
                bpm.append(row[0])

    return syllable_durations, bpm

def generatePinyin(scoreFilename):

    syllables           = []
    pinyins             = []
    syllable_durations  = []
    bpm                 = []
    try:
        with open(scoreFilename, 'rb') as csvfile:
            score = csv.reader(csvfile)
            for idx, row in enumerate(score):
                if (idx+1)%2:
                    syllables.append(row[1:])

                    row_pinyin = []
                    for r in row[1:]:
                        if len(r):
                            row_pinyin.append(pinyin.get(r, format="strip", delimiter=" "))
                        else:
                            row_pinyin.append('')
                    pinyins.append(row_pinyin)

                elif idx%2:
                    syllable_durations.append(row[1:])
                    bpm.append(row[0])
    except IOError:
        print scoreFilename, 'not found.'

    return syllables,pinyins,syllable_durations,bpm

def writerowCsv(syllables,syllable_durations,bpm,writer,pinyins=None):
    for ii in range(len(syllables)):
        writer.writerow(['']+syllables[ii])
        if pinyins:
            writer.writerow(['']+pinyins[ii])
        writer.writerow([bpm[ii]]+syllable_durations[ii])

def writeCsv(scoreFilename, syllables, syllable_durations, bpm):
    """
    write csv scores
    :param scoreFilename:
    :param syllables:
    :param syllable_durations:
    :param bpm:
    :return:
    """
    if len(syllables):

        directory, _ = os.path.split(scoreFilename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        export=open(scoreFilename, "wb")
        writer=csv.writer(export, delimiter=',')
        writerowCsv(syllables,syllable_durations,bpm,writer,None)
        export.close()

def writeCsvPinyin(scoreFilename, scoreFilenamePinyin):
    '''
    use this function to add pinyin to scoreFilename
    :param scoreFilename:
    :param scoreFilenamePinyin:
    :return:
    '''
    syllables,pinyins,syllable_durations,bpm = generatePinyin(scoreFilename)
    if len(syllables):
        export=open(scoreFilenamePinyin, "wb")
        writer=csv.writer(export, delimiter=',')
        writerowCsv(syllables,syllable_durations,bpm,writer,pinyins)
        export.close()