# -*- coding: utf-8 -*-
import sys, os, csv
import numpy as np

sys.path.append(os.path.realpath('./src/'))

import textgridParser
import labParser
import scoreParser
import evaluation


def batch_eval(aCapella_root, dataset_path, annotation_path, segPhrase_path, segSyllable_path, score_path, recordings, tolerance):

    sumDetectedBoundaries, sumGroundtruthPhrases, sumGroundtruthBoundaries, sumCorrect, sumOnsetCorrect, \
    sumOffsetCorrect, sumInsertion, sumDeletion = 0 ,0 ,0 ,0 ,0 ,0, 0, 0

    for i_recording, recording_name in enumerate(recordings):

        groundtruth_textgrid_file   = os.path.join(aCapella_root, dataset_path, annotation_path, recording_name+'.TextGrid')
        phrase_boundary_lab_file    = os.path.join(aCapella_root, dataset_path, segPhrase_path,  recording_name+'.lab')
        detected_lab_file_head      = os.path.join(aCapella_root, dataset_path, segSyllable_path,recording_name)
        score_file                  = os.path.join(aCapella_root, dataset_path, score_path,      recording_name+'.csv')

        if not os.path.isfile(score_file):
            print 'Score not found: ' + score_file
            continue

        lineList                    = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
        utteranceList               = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

        # parse lines of groundtruth
        nestedUtteranceLists, numLines, numUtterances   = textgridParser.wordListsParseByLines(lineList, utteranceList)

        # parse score
        utterance_durations, bpm                        = scoreParser.csvDurationScoreParser(score_file)


        # syllable boundaries groundtruth of each line
        for idx, list in enumerate(nestedUtteranceLists):
            if int(bpm[idx]):
                print 'Evaluating... ' + recording_name + ' phrase ' + str(idx+1)

                ul = list[1]
                firstStartTime          = ul[0][0]
                groundtruthBoundaries   = [(np.array(ul_element[:2]) - firstStartTime).tolist() for ul_element in ul]

                detected_syllable_lab   = detected_lab_file_head+'_'+str(idx+1)+'.syll.lab'
                if not os.path.isfile(detected_syllable_lab):
                    print 'Syll lab file not found: ' + detected_syllable_lab
                    continue

                # read boundary detected lab into python list
                detectedBoundaries          = labParser.lab2WordList(detected_syllable_lab)

                # read boundary groundtruth textgrid into python list

                # for segment in utteranceList:
                #     asciiLine = segment[2].encode("ascii", "replace")
                #     if len(asciiLine.replace(" ", "")):
                #         groundtruthBoundaries.append(segment[0:2])
                #
                # print groundtruthBoundaries

                #
                numDetectedBoundaries, numGroundtruthBoundaries, numCorrect, numOnsetCorrect, numOffsetCorrect, \
                numInsertion, numDeletion = evaluation.boundaryEval(groundtruthBoundaries, detectedBoundaries, tolerance)

                sumDetectedBoundaries       += numDetectedBoundaries
                sumGroundtruthBoundaries    += numGroundtruthBoundaries
                sumGroundtruthPhrases       += 1
                sumCorrect                  += numCorrect
                sumOnsetCorrect             += numOnsetCorrect
                sumOffsetCorrect            += numOffsetCorrect
                sumInsertion                += numInsertion
                sumDeletion                 += numDeletion

                if numCorrect/float(numGroundtruthBoundaries) < 0.7:
                    print "Detected: {0}, Ground truth: {1}, Correct: {2}, Onset correct: {3}, " \
                          "Offset correct: {4}, Insertion: {5}, Deletion: {6}\n".\
                        format(numDetectedBoundaries, numGroundtruthBoundaries,numCorrect, numOnsetCorrect,
                               numOffsetCorrect, numInsertion, numDeletion)

    return sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, \
           sumOffsetCorrect, sumInsertion, sumDeletion


def stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect,
             sumInsertion, sumDeletion, DB, GB, GP, C, OnC, OffC, I, D):
    return sumDetectedBoundaries+DB, sumGroundtruthBoundaries+GB, sumGroundtruthPhrases+GP, sumCorrect+C, \
           sumOnsetCorrect+OnC, sumOffsetCorrect+OffC, sumInsertion+I,sumDeletion+D


def evaluation_whole_dataset(segSyllable_path):

    ############################
    #       Initialisation     #
    ############################

    # aCapella root
    aCapella_root   = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/'

    # dataset path
    queenMarydataset_path    = 'QueenMary/jingjuSingingMono'
    londonRecording_path     = 'londonRecording'
    bcnRecording_path        = 'bcnRecording'

    segPhrase_path      = 'segPhrase'
    annotation_path     = 'annotation'
    score_path          = 'scoreDianSilence'


    # queen mary recordings
    queenMaryFem_01_Recordings = ['fem_01/neg_1', 'fem_01/neg_3', 'fem_01/neg_5', 'fem_01/pos_1', 'fem_01/pos_3', 'fem_01/pos_5', 'fem_01/pos_7']
    queenMaryFem_10_Recordings = ['fem_10/pos_1', 'fem_10/pos_3']
    queenMaryFem_11_Recordings = ['fem_11/pos_1']

    queenMaryMale_01_Recordings = ['male_01/neg_1','male_01/neg_2','male_01/neg_3','male_01/neg_4','male_01/neg_5',
                                'male_01/pos_1','male_01/pos_2','male_01/pos_3','male_01/pos_4','male_01/pos_5','male_01/pos_6']
    queenMaryMale_02_Recordings = ['male_02/neg_1']
    queenMaryMale_12_Recordings = ['male_12/neg_1']
    queenMaryMale_13_Recordings = ['male_13/pos_1', 'male_13/pos_3']

    queenMaryFem_Recordings     = queenMaryFem_01_Recordings + queenMaryFem_10_Recordings + queenMaryFem_11_Recordings
    queenMaryMale_Recordings    = queenMaryMale_01_Recordings + queenMaryMale_02_Recordings + queenMaryMale_12_Recordings + queenMaryMale_13_Recordings

    queenMary_Recordings        = queenMaryFem_Recordings + queenMaryMale_Recordings

    # london recordings
    londonDan_Recordings        = ['Dan-01', 'Dan-02', 'Dan-03', 'Dan-04']
    londonLaosheng_Recordings   = ['Laosheng-01', 'Laosheng-02', 'Laosheng-03', 'Laosheng-04']

    london_Recordings           = londonDan_Recordings + londonLaosheng_Recordings

    # bcn recordings
    bcnDan_Recordings           = ['001', '007']
    bcnLaosheng_Recordings      = ['003', '004', '005', '008']

    bcn_Recordings              = bcnDan_Recordings + bcnLaosheng_Recordings

    tolerance                   = 0.3


    ############################
    #       Evaluation         #
    ############################

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = 0 ,0 ,0 ,0 ,0 ,0, 0, 0

    # -- Female
    # queen mary
    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(aCapella_root, queenMarydataset_path, annotation_path, segPhrase_path,
                                            segSyllable_path, score_path, queenMaryFem_Recordings, tolerance)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases,sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion, DB, GB, GP, C, OnC, OffC, I, D)

    # london
    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(aCapella_root, londonRecording_path, annotation_path, segPhrase_path,
                                            segSyllable_path, score_path, londonDan_Recordings, tolerance)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion, DB, GB, GP, C, OnC, OffC, I, D)

    # bcn
    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(aCapella_root, bcnRecording_path, annotation_path, segPhrase_path,
                                            segSyllable_path, score_path, bcnDan_Recordings, tolerance)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion, DB, GB, GP, C, OnC, OffC, I, D)

    print "Detected: {0}, Ground truth: {1}, Ground truth phrases: {2} Correct rate: {3}, Insertion rate: {4}, Deletion rate: {5}\n".\
        format(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect/float(sumGroundtruthBoundaries),
           sumInsertion/float(sumGroundtruthBoundaries), sumDeletion/float(sumGroundtruthBoundaries))

    # -- Male
    # queen mary
    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(aCapella_root, queenMarydataset_path, annotation_path, segPhrase_path,
                                            segSyllable_path, score_path, queenMaryMale_Recordings, tolerance)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases,sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect,sumInsertion, sumDeletion, DB, GB, GP, C, OnC, OffC, I, D)

    # london
    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(aCapella_root, londonRecording_path, annotation_path, segPhrase_path,
                                            segSyllable_path, score_path, londonLaosheng_Recordings, tolerance)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion, DB, GB, GP, C, OnC, OffC, I, D)

    # bcn
    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(aCapella_root, bcnRecording_path, annotation_path, segPhrase_path,
                                            segSyllable_path, score_path, bcnLaosheng_Recordings, tolerance)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion, DB, GB, GP, C, OnC, OffC, I, D)

    #print sumDetectedBoundaries, sumGroundtruthBoundaries, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion

    print "Detected: {0}, Ground truth: {1}, Ground truth phrases: {2} Correct rate: {3}, Insertion rate: {4}, Deletion rate: {5}\n".\
        format(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect/float(sumGroundtruthBoundaries),
           sumInsertion/float(sumGroundtruthBoundaries), sumDeletion/float(sumGroundtruthBoundaries))

    return sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect/float(sumGroundtruthBoundaries), \
           sumInsertion/float(sumGroundtruthBoundaries), sumDeletion/float(sumGroundtruthBoundaries)

################################
#       standard deviation     #
################################
# eval_result_file_name       = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/eval_result.csv'
# sdevs                       = np.arange(0.05,1,0.05)
# dev_modes                   = ['constant', 'proportion']
#
# with open(eval_result_file_name, 'w') as testfile:
#     csv_writer = csv.writer(testfile)
#
#     for dm in dev_modes:
#
#         for sdev in sdevs:
#
#             segSyllable_path    = 'segSyllable/viterbiSilenceWeighting/segSyllable_rong_' + dm + '_' + str(sdev)
#             detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate = \
#                 evaluation_whole_dataset(segSyllable_path)
#
#             csv_writer.writerow([dm,sdev,detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate])

################################
#       vad duration           #
################################

# eval_result_file_name       = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/eval_result.csv'
# vadDurs                     = np.arange(0.01,0.09,0.01)
#
# with open(eval_result_file_name, 'w') as testfile:
#     csv_writer = csv.writer(testfile)
#
#     for vadDur in vadDurs:
#
#         segSyllable_path    = 'segSyllable/vadDuration/segSyllable_rong_proportion' + '_' + str(0.35) + '_' + str(vadDur)
#         detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate = \
#             evaluation_whole_dataset(segSyllable_path)
#
#         csv_writer.writerow([vadDur,detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate])

################################
#       vad weighting          #
################################

# eval_result_file_name       = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/eval_result_vadWeight.csv'
# vadWeights                     = np.arange(0.1,1.0,0.1)
#
# with open(eval_result_file_name, 'w') as testfile:
#     csv_writer = csv.writer(testfile)
#
#     for vadWeight in vadWeights:
#
#         segSyllable_path    = 'segSyllable/vadWeighting/segSyllable_rong_proportion' + '_' + str(0.35) + '_' + str(vadWeight)
#         detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate = \
#             evaluation_whole_dataset(segSyllable_path)
#
        # csv_writer.writerow([vadWeight,detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate])

################################
#      nicolas mean            #
################################

# eval_result_file_name       = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/eval_result_nicolas.csv'
# duration_means              = np.arange(0.1,1.0,0.1)
#
# with open(eval_result_file_name, 'w') as testfile:
#     csv_writer = csv.writer(testfile)
#
#     for dm in duration_means:
#
#         segSyllable_path    = 'segSyllable_nicolas/segSyllable_nicolas' + '_' + str(dm)
#         detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate = \
#             evaluation_whole_dataset(segSyllable_path)
#
#         csv_writer.writerow([dm,detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate])


# segSyllable_path    = 'segSyllable/viterbiSilenceWeighting/segSyllable_rong_' + 'proportion' + '_' + str(0.35) + '_' +
# detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate = \
#     evaluation_whole_dataset(segSyllable_path)

segSyllable_path    = 'segSyllable/vadWeighting/segSyllable_rong_proportion' + '_' + str(0.35) + '_' + str(1)
detected, ground_truth, ground_truth_phrases, correct_rate, insertion_rate, deletion_rate = \
    evaluation_whole_dataset(segSyllable_path)

