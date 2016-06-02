import georgiParser
import textgridParser
import evaluation
import os

def g_eval(groundtruth_textgrid_filename, georgi_alignment_filename, tolerance):

    boundaryList                = georgiParser.syllables_total_parser(georgi_alignment_filename)

    try:

        utteranceList               = textgridParser.textGrid2WordList(groundtruth_textgrid_filename, whichTier='dian', utf16=False)

        utteranceDuration           = textgridParser.textGrid2WordList(groundtruth_textgrid_filename, whichTier='dianDuration', utf16=False)

    except:
        utteranceList               = textgridParser.textGrid2WordList(groundtruth_textgrid_filename, whichTier='dian', utf16=True)

        utteranceDuration           = textgridParser.textGrid2WordList(groundtruth_textgrid_filename, whichTier='dianDuration', utf16=True)

    # remove empty dian
    tempGroundtruthList         = []
    groundtruthDuration         = []
    for idx, utterance in enumerate(utteranceList):
        if len(utterance[2].strip()):# and float(utteranceDuration[idx][2]):
            tempGroundtruthList.append(utterance)
            groundtruthDuration.append(float(utteranceDuration[idx][2]))

    # remove 0 duration
    detectedBoundaryList    = []
    groundtruthList         = []
    for idx, utterance in enumerate(tempGroundtruthList):
        if groundtruthDuration[idx]:
            groundtruthList.append(tempGroundtruthList[idx])
            detectedBoundaryList.append(boundaryList[idx])

    numDetectedBoundaries, numGroundtruthBoundaries, numCorrect, numOnsetCorrect, numOffsetCorrect, numInsertion, numDeletion = \
        evaluation.boundaryEval(groundtruthList,detectedBoundaryList,tolerance)

    print "Detected: {0}, Ground truth: {1}, Correct: {2}, Onset correct: {3}, " \
                              "Offset correct: {4}, Insertion: {5}, Deletion: {6}\n".\
                            format(numDetectedBoundaries, numGroundtruthBoundaries,numCorrect, numOnsetCorrect,
                                   numOffsetCorrect, numInsertion, numDeletion)

    return  numDetectedBoundaries, numGroundtruthBoundaries, numCorrect, numOnsetCorrect, numOffsetCorrect, numInsertion, numDeletion

def result_output(folders,tolerance):
    sumDetectedBoundaries, sumGroundtruthBoundaries, sumCorrect, sumOnsetCorrect, \
               sumOffsetCorrect, sumInsertion, sumDeletion = 0,0,0,0,0,0,0

    for folder in folders:
        filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for filename in filenames:
            extension = os.path.splitext(filename)[1]
            if extension == '.TextGrid':
                filenameRoot = os.path.splitext(filename)[0]
                georgi_alignment_filename       = os.path.join(folder, filenameRoot + '.syllables_total_dev_0.5')
                groundtruth_textgrid_filename   = os.path.join(folder, filename)

                print groundtruth_textgrid_filename

                if os.path.isfile(georgi_alignment_filename) and os.path.isfile(groundtruth_textgrid_filename):

                    numDetectedBoundaries, numGroundtruthBoundaries, numCorrect, numOnsetCorrect,\
                    numOffsetCorrect, numInsertion, numDeletion \
                        = g_eval(groundtruth_textgrid_filename, georgi_alignment_filename, tolerance)

                    sumDetectedBoundaries       += numDetectedBoundaries
                    sumGroundtruthBoundaries    += numGroundtruthBoundaries
                    sumCorrect                  += numCorrect
                    sumOnsetCorrect             += numOnsetCorrect
                    sumOffsetCorrect            += numOffsetCorrect
                    sumInsertion                += numInsertion
                    sumDeletion                 += numDeletion

    print "Detected: {0}, Ground truth: {1}, Correct rate: {2}, Insertion rate: {3}, Deletion rate: {4}\n".\
            format(sumDetectedBoundaries, sumGroundtruthBoundaries, sumCorrect/float(sumGroundtruthBoundaries),
               sumInsertion/float(sumGroundtruthBoundaries), sumDeletion/float(sumGroundtruthBoundaries))

# georgi_alignment_file               = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/testFiles/georgi/rong_female_neg_1.syllables_total_dev_3.0'
# groundtruth_textgrid_file           = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/QueenMary/jingjuSingingMono/annotation/fem_01/neg_1.TextGrid'
# georgi_alignment_file               = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/testFiles/georgi/rong_male_neg_1.syllables_total_dev_3.0'
# groundtruth_textgrid_file           = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/QueenMary/jingjuSingingMono/annotation/male_01/neg_1.TextGrid'

tolerance                           = 0.3

georgi_folder_male                  = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/georgi/male'

georgi_subfolders_male              = [x[0] for x in os.walk(georgi_folder_male)]

georgi_folder_female                = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/georgi/female'

georgi_subfolders_female            = [x[0] for x in os.walk(georgi_folder_female)]

result_output(georgi_subfolders_male,tolerance)

# result_output(georgi_subfolders_female,tolerance)