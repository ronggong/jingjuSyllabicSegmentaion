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
from src.trainTestSeparation import getTestTrainRecordingsMaleFemale, \
    getTestTrainrecordingsRiyaz, \
    getTestTrainRecordingsNactaISMIR, \
    getTestTrainRecordingsArtist, \
    getTestTrainRecordingsArtistAlbumFilter


def batch_eval(root_path, annotation_path, segPhrase_path, segSyllable_path, score_path, groundtruth_path, eval_details_path, recordings, tolerance, method='obin', label=False):

    sumDetectedBoundaries, sumGroundtruthPhrases, sumGroundtruthBoundaries, sumCorrect, sumOnsetCorrect, \
    sumOffsetCorrect, sumInsertion, sumDeletion = 0 ,0 ,0 ,0 ,0 ,0, 0, 0

    for artist_path, recording_name in recordings:

        if annotation_path:
            groundtruth_textgrid_file   = os.path.join(annotation_path, artist_path, recording_name+'.TextGrid')
            groundtruth_lab_file_head = os.path.join(groundtruth_path, artist_path)
        else:
            groundtruth_syllable_lab   = os.path.join(groundtruth_path, artist_path, recording_name+'.lab')


        # phrase_boundary_lab_file    = os.path.join(segPhrase_path, artist_path,  recording_name+'.lab')
        if method == 'obin':
            # syll-o-matic output
            detected_lab_file_head      = os.path.join(root_path, segSyllable_path, artist_path, recording_name)
        else:
            # jan output
            detected_lab_file_head      = os.path.join(segSyllable_path, artist_path,recording_name)

        score_file                  = os.path.join(score_path, artist_path,  recording_name+'.csv')
        # parse score
        _, utterance_durations, bpm = scoreParser.csvDurationScoreParser(score_file)

        if eval_details_path:
            eval_result_details_file_head = os.path.join(eval_details_path, artist_path)

        if not os.path.isfile(score_file):
            print 'Score not found: ' + score_file
            continue

        if annotation_path:
            # create ground truth lab path, if not exist
            if not os.path.isdir(groundtruth_lab_file_head):
                os.makedirs(groundtruth_lab_file_head)

            if not os.path.isdir(eval_result_details_file_head):
                os.makedirs(eval_result_details_file_head)

            lineList                    = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
            utteranceList               = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

            # parse lines of groundtruth
            nestedUtteranceLists, numLines, numUtterances   = textgridParser.wordListsParseByLines(lineList, utteranceList)

            # create the ground truth lab files
            for idx,list in enumerate(nestedUtteranceLists):
                try:
                    print(bpm[idx])
                except IndexError:
                    continue

                if float(bpm[idx]):
                    print 'Creating ground truth lab ... ' + recording_name + ' phrase ' + str(idx+1)

                    ul = list[1]
                    firstStartTime          = ul[0][0]
                    groundtruthBoundaries   = [(np.array(ul_element[:2]) - firstStartTime).tolist() + [ul_element[2]] for ul_element in ul]
                    groundtruth_syllable_lab   = join(groundtruth_lab_file_head, recording_name+'_'+str(idx+1)+'.syll.lab')

                    with open(groundtruth_syllable_lab, "wb") as text_file:
                        for gtbs in groundtruthBoundaries:
                            text_file.write("{0} {1} {2}\n".format(gtbs[0],gtbs[1],gtbs[2]))
        else:
            nestedUtteranceLists = [labParser.lab2WordList(groundtruth_syllable_lab, label=True)]

        # syllable boundaries groundtruth of each line
        # ignore eval details
        # eval_details_csv    = join(eval_result_details_file_head, recording_name+'.csv')
        # with open(eval_details_csv, 'wb') as csv_file:
        #     csv_writer = csv.writer(csv_file)

        for idx, list in enumerate(nestedUtteranceLists):
            try:
                print(bpm[idx])
            except IndexError:
                continue

            if float(bpm[idx]):
                print 'Evaluating... ' + recording_name + ' phrase ' + str(idx+1)

                if annotation_path:
                    ul = list[1]
                    firstStartTime          = ul[0][0]
                    groundtruthBoundaries   = [(np.array(ul_element[:2]) - firstStartTime).tolist() + [ul_element[2]] for ul_element in ul]
                else:
                    firstStartTime          = list[0][0]
                    groundtruthBoundaries   = [(np.array(ul_element[:2]) - firstStartTime).tolist() + [ul_element[2]]for ul_element in list]

                detected_syllable_lab   = detected_lab_file_head+'_'+str(idx+1)+'.syll.lab'
                if not os.path.isfile(detected_syllable_lab):
                    print 'Syll lab file not found: ' + detected_syllable_lab
                    continue

                # read boundary detected lab into python list
                detectedBoundaries          = labParser.lab2WordList(detected_syllable_lab, label=label)

                # detectedBoundaries = [[d[0]*44100/16000.0, d[1]*44100/16000.0] for d in detectedBoundaries]

                print(groundtruthBoundaries)
                print(groundtruth_syllable_lab)
                print(detectedBoundaries)
                print(detected_syllable_lab)
                #
                numDetectedBoundaries, numGroundtruthBoundaries, numCorrect, numOnsetCorrect, numOffsetCorrect, \
                numInsertion, numDeletion, correct_list = evaluation2.boundaryEval(groundtruthBoundaries, detectedBoundaries, tolerance)

                sumDetectedBoundaries       += numDetectedBoundaries
                sumGroundtruthBoundaries    += numGroundtruthBoundaries
                sumGroundtruthPhrases       += 1
                sumCorrect                  += numCorrect
                sumOnsetCorrect             += numOnsetCorrect
                sumOffsetCorrect            += numOffsetCorrect
                sumInsertion                += numInsertion
                sumDeletion                 += numDeletion

                # if numCorrect/float(numGroundtruthBoundaries) < 0.7:
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

def evaluation_test_dataset(segSyllablePath, tolerance, method):

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = 0, 0, 0, 0, 0, 0, 0, 0

    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtistAlbumFilter()

    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(nacta2017_dataset_root_path, nacta2017_textgrid_path,nacta2017_segPhrase_path,
                                                segSyllablePath, nacta2017_score_path,
                                                nacta2017_groundtruthlab_path, nacta2017_eval_details_path,
                                                testNacta2017, tolerance, method, True)

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases,
                                         sumCorrect,
                                         sumOnsetCorrect, sumOffsetCorrect, sumInsertion, sumDeletion, DB, GB, GP, C,
                                         OnC, OffC, I, D)

    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(nacta_dataset_root_path, nacta_textgrid_path, nacta_segPhrase_path,
                                                segSyllablePath, nacta_score_path,
                                                nacta_groundtruthlab_path, nacta_eval_details_path,
                                                testNacta, tolerance, method, label=True)

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

def evaluation_riyaz_test_dataset(segSyllablePath, tolerance, method, label):

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = 0, 0, 0, 0, 0, 0, 0, 0

    testRiyaz, trainRiyaz = getTestTrainrecordingsRiyaz()

    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(riyaz_dataset_root_path, None, None,
                                                segSyllablePath, riyaz_score_path,
                                                riyaz_groundtruthlab_path, None,
                                                testRiyaz, tolerance, method, label=label)

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
    eval_result_file_name       = './eval/results/jan_old+new_artist_filter_split/eval_result_jan_class_weight_label.csv'
    segSyllable_path            = './eval/results/jan_old+new_artist_filter_split'
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
            eval_result_file_name       = './eval/results/jordi_3_fuison_old+new/eval_result_jordi_class_weight_conv_dense_horizontal_timbral_filter_win.csv'
            segSyllable_path            = './eval/results/jordi_3_fuison_old+new'
    else:
        if filter_shape == 'temporal':
            if layer2 == 20:
                eval_result_file_name       = './eval/results/jordi_cw_conv_dense_layer2_20_win/eval_result_jordi_class_weight_conv_dense_win.csv'
                segSyllable_path            = './eval/results/jordi_cw_conv_dense_layer2_20_win'
            else:
                # layer2 32 nodes
                eval_result_file_name       = './eval/results/jordi_temporal_old+new_artist_filter_split/eval_result_jordi_class_weight_conv_dense_win_label.csv'
                segSyllable_path            = './eval/results/jordi_temporal_old+new_artist_filter_split'
        else:
            # timbral filter shape
            if layer2 == 20:
                eval_result_file_name       = './eval/results/jordi_cw_conv_dense_timbral_filter_layer2_20_win/eval_result_jordi_class_weight_conv_dense_timbral_filter_win.csv'
                segSyllable_path            = './eval/results/jordi_cw_conv_dense_timbral_filter_layer2_20_win'
            else:
                # layer2 32 nodes
                eval_result_file_name       = './eval/results/jordi_timbral_old+new_artist_filter_split/eval_result_jordi_class_weight_conv_dense_timbral_filter_win_label.csv'
                segSyllable_path            = './eval/results/jordi_timbral_old+new_artist_filter_split'

print(eval_result_file_name)
print(segSyllable_path)
tols                = [0.025,0.05,0.1,0.15,0.2,0.25,0.3]
with open(eval_result_file_name, 'wb') as testfile:
    csv_writer = csv.writer(testfile)
    for t in tols:
        detected, ground_truth, ground_truth_phrases, correct, insertion, deletion = \
            evaluation_test_dataset(segSyllable_path,tolerance=t, method='jan')
        recall,precision,F1 = evaluation2.metrics(detected,ground_truth,correct)
        print(detected, ground_truth, correct)
        print(recall, precision, F1)
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

# # vad
# eval_result_file_name       = './eval/results/obin_riyaz_syllables/eval_result_obin_test.csv'
# segSyllable_path            = 'segSyllable_obin_proportion' + '_' + str(0.35) + '_' + str(0.2)
#
# # no vad
# # eval_result_file_name       = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/eval_result_rong_novad_different_tolerence_test.csv'
# # segSyllable_path            = 'segSyllable/vadnoAperiocityWeighting/segSyllable_rong_proportion' + '_' + str(0.35) + '_' + str(1)
#
# # tols                = [0.025, 0.05,0.1,0.15,0.2,0.25,0.3]
# tols = [0.05]
# with open(eval_result_file_name, 'wb') as testfile:
#     csv_writer = csv.writer(testfile)
#     for t in tols:
#         detected, ground_truth, ground_truth_phrases, correct, insertion, deletion = \
#             evaluation_riyaz_test_dataset(segSyllable_path,tolerance=t,method='obin',label=True)
#         recall,precision,F1 = evaluation2.metrics(detected,ground_truth,correct)
#         csv_writer.writerow([t,detected, ground_truth, ground_truth_phrases, recall,precision,F1])