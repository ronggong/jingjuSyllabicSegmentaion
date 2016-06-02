import csv

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