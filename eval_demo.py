# -*- coding: utf-8 -*-
import sys, os, csv
import numpy as np

sys.path.append(os.path.realpath('./src/'))

import textgridParser
import labParser
import scoreParser
# import evaluation
import evaluation2
from filePath import *


def batch_eval(aCapella_root, dataset_path, annotation_path, segPhrase_path, segSyllable_path, score_path, recordings, tolerance, label=True):

    sumDetectedBoundaries, sumGroundtruthPhrases, sumGroundtruthBoundaries, sumCorrect, sumOnsetCorrect, \
    sumOffsetCorrect, sumInsertion, sumDeletion = 0 ,0 ,0 ,0 ,0 ,0, 0, 0

    for i_recording, recording_name in enumerate(recordings):

        groundtruth_textgrid_file   = os.path.join(aCapella_root, dataset_path, annotation_path, recording_name+'.TextGrid')
        phrase_boundary_lab_file    = os.path.join(aCapella_root, dataset_path, segPhrase_path,  recording_name+'.lab')
        # syll-o-matic output
        # detected_lab_file_head      = os.path.join(aCapella_root, dataset_path, segSyllable_path,recording_name)
        # jan output
        detected_lab_file_head      = os.path.join(segSyllable_path, dataset_path,recording_name)

        score_file                  = os.path.join(aCapella_root, dataset_path, score_path,      recording_name+'.csv')

        groundtruth_lab_file_head   = os.path.join(aCapella_root, dataset_path, groundtruth_lab_path, recording_name)
        eval_result_details_file_head = os.path.join(aCapella_root, dataset_path, eval_details_path, recording_name)

        if not os.path.isfile(score_file):
            print 'Score not found: ' + score_file
            continue

        # create ground truth lab path, if not exist
        if not os.path.isdir(groundtruth_lab_file_head):
            os.makedirs(groundtruth_lab_file_head)

        if not os.path.isdir(eval_result_details_file_head):
            os.makedirs(eval_result_details_file_head)

        lineList                    = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
        utteranceList               = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

        # parse lines of groundtruth
        nestedUtteranceLists, numLines, numUtterances   = textgridParser.wordListsParseByLines(lineList, utteranceList)

        # parse score
        utterance_durations, bpm                        = scoreParser.csvDurationScoreParser(score_file)

        # create the ground truth lab files
        for idx,list in enumerate(nestedUtteranceLists):
            if int(bpm[idx]):
                print 'Creating ground truth lab ... ' + recording_name + ' phrase ' + str(idx+1)

                ul = list[1]
                firstStartTime          = ul[0][0]
                groundtruthBoundaries   = [(np.array(ul_element[:2]) - firstStartTime).tolist() + [ul_element[2]] for ul_element in ul]
                groundtruth_syllable_lab   = groundtruth_lab_file_head+'_'+str(idx+1)+'.syll.lab'

                with open(groundtruth_syllable_lab, "wb") as text_file:
                    for gtbs in groundtruthBoundaries:
                        text_file.write("{0} {1} {2}\n".format(gtbs[0],gtbs[1],gtbs[2]))

        # syllable boundaries groundtruth of each line
        # eval_details_csv    = eval_result_details_file_head+'.csv'
        # with open(eval_details_csv, 'wb') as csv_file:
        #     csv_writer = csv.writer(csv_file)

        for idx, list in enumerate(nestedUtteranceLists):
            if int(bpm[idx]):
                print 'Evaluating... ' + recording_name + ' phrase ' + str(idx+1)

                ul = list[1]
                firstStartTime          = ul[0][0]
                groundtruthBoundaries   = [(np.array(ul_element[:2]) - firstStartTime).tolist() + [ul_element[2]] for ul_element in ul]

                detected_syllable_lab   = detected_lab_file_head+'_'+str(idx+1)+'.syll.lab'
                if not os.path.isfile(detected_syllable_lab):
                    print 'Syll lab file not found: ' + detected_syllable_lab
                    continue

                # read boundary detected lab into python list
                detectedBoundaries          = labParser.lab2WordList(detected_syllable_lab, withLabel=label)

                #
                numDetectedBoundaries, numGroundtruthBoundaries, numCorrect, numOnsetCorrect, numOffsetCorrect, \
                numInsertion, numDeletion, correct_list = evaluation2.boundaryEval(groundtruthBoundaries, detectedBoundaries, tolerance, label)

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

                    # csv_writer.writerow([recording_name+'_'+str(idx+1),
                    #                      numDetectedBoundaries,
                    #                      numGroundtruthBoundaries,
                    #                      numCorrect,
                    #                      numInsertion,
                    #                      numDeletion,
                    #                      correct_list])

    return sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, \
           sumOffsetCorrect, sumInsertion, sumDeletion


def stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect,
             sumInsertion, sumDeletion, DB, GB, GP, C, OnC, OffC, I, D):
    return sumDetectedBoundaries+DB, sumGroundtruthBoundaries+GB, sumGroundtruthPhrases+GP, sumCorrect+C, \
           sumOnsetCorrect+OnC, sumOffsetCorrect+OffC, sumInsertion+I,sumDeletion+D

####---- function to insert pinyin into duration csv files
def batch_insert_pinyin_2_csv(aCapella_root, dataset_path, score_path, recordings):
    for i_recording, recording_name in enumerate(recordings):
        score_file                  = os.path.join(aCapella_root, dataset_path, score_path, recording_name+'.csv')
        score_file_pinyin           = os.path.join(aCapella_root, dataset_path, score_path, recording_name+'_pinyin.csv')

        scoreParser.writeCsvPinyin(score_file,score_file_pinyin)

def insert_pinyin_2_csv_whole_dataset():

    batch_insert_pinyin_2_csv(aCapella_root,queenMarydataset_path,score_path,queenMary_Recordings)

    batch_insert_pinyin_2_csv(aCapella_root,londonRecording_path,score_path,london_Recordings)

    batch_insert_pinyin_2_csv(aCapella_root,bcnRecording_path,score_path,bcn_Recordings)


def evaluation_whole_dataset(segSyllable_path,tolerance):

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

    return sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumInsertion, sumDeletion

def evaluation_test_dataset(segSyllablePath, tolerance, label):

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = 0, 0, 0, 0, 0, 0, 0, 0

    # queen mary
    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(aCapella_root, queenMarydataset_path, annotation_path, segPhrase_path,
                                                segSyllable_path, score_path, queenMary_Recordings_test, tolerance, label)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases,
                                         sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion, DB, GB, GP, C,
                                         OnC, OffC, I, D)

    # london
    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(aCapella_root, londonRecording_path, annotation_path, segPhrase_path,
                                                segSyllable_path, score_path, london_Recordings_test, tolerance, label)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases,
                                         sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion, DB, GB, GP, C,
                                         OnC, OffC, I, D)

    # bcn
    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(aCapella_root, bcnRecording_path, annotation_path, segPhrase_path,
                                                segSyllable_path, score_path, bcn_Recordings_test, tolerance, label)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases,
                                         sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion, DB, GB, GP, C,
                                         OnC, OffC, I, D)

    print "Detected: {0}, Ground truth: {1}, Ground truth phrases: {2} Correct rate: {3}, Insertion rate: {4}, Deletion rate: {5}\n". \
        format(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases,
               sumCorrect / float(sumGroundtruthBoundaries),
               sumInsertion / float(sumGroundtruthBoundaries), sumDeletion / float(sumGroundtruthBoundaries))

    return sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumInsertion, sumDeletion


############################################
#       jan jordi class weight             #
############################################

if mth_ODF == 'jan':
    eval_result_file_name       = './eval/results/jan_deep_old_ismir_win_peakPicking/eval_result_jan_class_weight.csv'
    segSyllable_path            = './eval/results/jan_deep_old_ismir_win_peakPicking'
elif mth_ODF == 'jan_chan3':
    eval_result_file_name       = './eval/results/jan_cw_3_chans_win/eval_result_jan_class_weight.csv'
    segSyllable_path            = './eval/results/jan_cw_3_chans_win'
elif mth_ODF == 'jordi_horizontal_timbral':
    if layer2 == 20:
        eval_result_file_name       = './eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_layer2_20_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
        segSyllable_path            = './eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_layer2_20_win'
    else:
        eval_result_file_name       = './eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
        segSyllable_path            = './eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_win'
else:
    # mth_ODF == 'jordi'
    if fusion:
        if layer2 == 20:
            eval_result_file_name       = './eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_multiply_layer2_20_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
            segSyllable_path            = './eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_multiply_layer2_20_win'
        else:
            eval_result_file_name       = './eval/results/jordi_fusion_old+new+ismir_split_win_peakPickingMadmom/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
            segSyllable_path            = './eval/results/jordi_fusion_old+new+ismir_split_win_peakPickingMadmom'
    else:
        if filter_shape == 'temporal':
            if layer2 == 20:
                eval_result_file_name       = './eval/results/jordi_cw_conv_dense_layer2_20_win/eval_result_jordi_class_weight_conv_dense_win.csv'
                segSyllable_path            = './eval/results/jordi_cw_conv_dense_layer2_20_win'
            else:
                # layer2 32 nodes
                eval_result_file_name       = './eval/results/jordi_temporal_old+new+ismir_split_win_peakPickingMadmom/eval_result_jordi_class_weight_conv_dense_win.csv'
                segSyllable_path            = './eval/results/jordi_temporal_old+new+ismir_split_win_peakPickingMadmom'
        else:
            # timbral filter shape
            if layer2 == 20:
                eval_result_file_name       = './eval/results/jordi_cw_conv_dense_timbral_filter_layer2_20_win/eval_result_jordi_class_weight_conv_dense_timbral_filter_win.csv'
                segSyllable_path            = './eval/results/jordi_cw_conv_dense_timbral_filter_layer2_20_win'
            else:
                # layer2 32 nodes
                eval_result_file_name       = './eval/results/jordi_timbral_old+new+ismir_split_win_peakPickingMadmom/eval_result_jordi_class_weight_conv_dense_timbral_filter_win.csv'
                segSyllable_path            = './eval/results/jordi_timbral_old+new+ismir_split_win_peakPickingMadmom'


tols                = [0.025,0.05,0.1,0.15,0.2,0.25,0.3]
# tols = [0.05]
with open(eval_result_file_name, 'wb') as testfile:
    csv_writer = csv.writer(testfile)
    for t in tols:
        detected, ground_truth, ground_truth_phrases, correct, insertion, deletion = \
            evaluation_test_dataset(segSyllable_path,tolerance=t, label=False)
        recall,precision,F1 = evaluation2.metrics(detected,ground_truth,correct)
        csv_writer.writerow([t,detected, ground_truth, ground_truth_phrases, recall,precision,F1])

# not used

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_cw/eval_result_jan_class_weight.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_cw'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_cw_3_chans_win/eval_result_jan_class_weight.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_cw_3_chans_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_cw_3_chans_layer1_70_win/eval_result_jan_class_weight.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_cw_3_chans_layer1_70_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_ncw/eval_result_jan_no_class_weight.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_ncw'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_win/eval_result_jordi_class_weight_conv_dense_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_149k_win/eval_result_jordi_class_weight_conv_dense_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_149k_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_layer2_20_win/eval_result_jordi_class_weight_conv_dense_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_layer2_20_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_timbral_filter_win/eval_result_jordi_class_weight_conv_dense_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_timbral_filter_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_timbral_filter_late_fusion_2_models_multiply_win/eval_result_jordi_class_weight_conv_dense_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_timbral_filter_late_fusion_2_models_multiply_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_timbral_filter_152k_win/eval_result_jordi_class_weight_conv_dense_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_timbral_filter_152k_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_timbral_filter_layer2_20_win/eval_result_jordi_class_weight_conv_dense_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_timbral_filter_layer2_20_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_layer2_20_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_layer2_20_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_layer2_20_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_layer2_20_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_multiply_layer2_20_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_multiply_layer2_20_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_multiply_coef_0.9_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_timbral_filter_late_fusion_multiply_coef_0.9_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_filter_late_fusion_2_models_multiply_win/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw_conv_dense_horizontal_filter_late_fusion_2_models_multiply_win'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw/eval_result_jordi_class_weight.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_cw'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_ncw/eval_result_jordi_no_class_weight.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jordi_ncw'

# eval_result_file_name       = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_cw_win/eval_result_jordi_class_weight_win.csv'
# segSyllable_path            = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/eval/results/jan_cw_win'

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

# eval_result_file_name       = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/eval_result_nicolas_different_tolerence.csv'
# duration_means              = np.arange(0.1,1.0,0.1)
#
# with open(eval_result_file_name, 'w') as testfile:
#     csv_writer = csv.writer(testfile)
#     for t in [0.05,0.1,0.15,0.2,0.25,0.3]:
#
#         # for dm in duration_means:
#         for dm in[0.7]:
#
#             segSyllable_path    = 'segSyllable_nicolas/segSyllable_nicolas' + '_' + str(dm)
#             detected, ground_truth, ground_truth_phrases, correct, insertion, deletion = \
#                 evaluation_whole_dataset(segSyllable_path,t)
#             recall,precision,F1 = evaluation2.metrics(detected,ground_truth,correct)
#             csv_writer.writerow([t,detected, ground_truth, ground_truth_phrases, recall,precision,F1])

################################
#      rong                    #
################################

# vad
# eval_result_file_name       = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/eval_result_rong_vad_different_tolerence_test.csv'
# segSyllable_path            = 'segSyllable/vadnoAperiocityWeighting/segSyllable_rong_proportion' + '_' + str(0.35) + '_' + str(0.2)

# no vad
# eval_result_file_name       = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/eval_result_rong_novad_different_tolerence_test.csv'
# segSyllable_path            = 'segSyllable/vadnoAperiocityWeighting/segSyllable_rong_proportion' + '_' + str(0.35) + '_' + str(1)
#
# tols                = [0.025, 0.05,0.1,0.15,0.2,0.25,0.3]
#
# with open(eval_result_file_name, 'wb') as testfile:
#     csv_writer = csv.writer(testfile)
#     for t in tols:
#         detected, ground_truth, ground_truth_phrases, correct, insertion, deletion = \
#             evaluation_test_dataset(segSyllable_path,tolerance=t)
#         recall,precision,F1 = evaluation2.metrics(detected,ground_truth,correct)
#         csv_writer.writerow([t,detected, ground_truth, ground_truth_phrases, recall,precision,F1])

