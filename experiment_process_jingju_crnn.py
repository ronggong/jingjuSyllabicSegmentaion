# -*- coding: utf-8 -*-
import pickle
from os import makedirs
from os.path import isfile, exists
from src.file_path_jingju_rnn import *
from src.labWriter import boundaryLabWriter
from src.utilFunctions import featureReshape
from src.utilFunctions import smooth_obs
from src.trainTestSeparation import getTestRecordingsScoreDurCorrectionArtistAlbumFilter

from madmom.features.onsets import OnsetPeakPickingProcessor
from audio_preprocessing import getMFCCBands2DMadmom
from training_scripts.models_CRNN import jan_original
from experiment_process_helper import data_parser
from experiment_process_helper import get_line_properties
from experiment_process_helper import get_results_decoding_path
from experiment_process_helper import boundary_decoding
from experiment_process_helper import get_boundary_list
from experiment_process_helper import writeResults2Txt
from eval_demo import eval_write_2_txt

from plot_code import plot_jingju

import numpy as np
import pyximport

pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

import viterbiDecoding


def batch_process_onset_detection(wav_path,
                                  textgrid_path,
                                  score_path,
                                  scaler,
                                  test_recordings,
                                  model_keras_cnn_0,
                                  cnnModel_name,
                                  eval_results_path,
                                  architecture=varin['architecture'],
                                  lab=False,
                                  threshold=0.54,
                                  obs_cal=True,
                                  decoding_method='viterbi',
                                  stateful=True):
    """
    experiment process, evaluate a whole jingju dataset using CRNN model
    :param wav_path: string, where we store the wav
    :param textgrid_path: string, where we store the textgrid
    :param score_path: string, where we store the score
    :param scaler: sklearn object, StandardScaler
    :param test_recordings: list of strings, testing recording filename
    :param model_keras_cnn_0: keras .h5, model weights
    :param cnnModel_name: string, model name
    :param eval_results_path: string, where we store the evaluation results
    :param architecture: string, model architecture name
    :param lab: string, for Riyaz dataset, not used in the paper
    :param threshold: float, threshold for peak picking onset selection
    :param obs_cal: string, tocal or toload, for saving running time
    :param decoding_method: string, viterbi or peakPicking
    :param stateful: bool, whether to use the stateful RNN
    :return:
    """

    eval_results_decoding_path = \
        get_results_decoding_path(decoding_method=decoding_method,
                                  bool_corrected_score_duration=varin['corrected_score_duration'],
                                  eval_results_path=eval_results_path)

    for artist_path, rn in test_recordings:

        score_file = join(score_path, artist_path, rn+'.csv')

        if not isfile(score_file):
            print 'Score not found: ' + score_file
            continue

        nested_syllable_lists, wav_file, line_list, syllables, syllable_durations, bpm, pinyins = \
            data_parser(artist_path=artist_path,
                        wav_path=wav_path,
                        textgrid_path=textgrid_path,
                        rn=rn,
                        score_file=score_file,
                        lab=lab)

        if obs_cal == 'tocal':
            # load audio
            mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
            mfcc_scaled = scaler.transform(mfcc)

        i_line = -1
        for i_obs, line in enumerate(line_list):
            if not lab and len(line[2]) == 0:
                continue

            i_line += 1

            try:
                print(syllable_durations[i_line])
            except IndexError:
                continue

            if float(bpm[i_line]) == 0:
                continue

            time_line, lyrics_line, frame_start, frame_end = get_line_properties(lab=lab,
                                                                                 line=line,
                                                                                 hopsize_t=hopsize_t)

            obs_path = join('./obs', cnnModel_name, artist_path)
            obs_filename = rn + '_' + str(i_line + 1) + '.pkl'

            if obs_cal == 'tocal':
                mfcc_line = mfcc[frame_start:frame_end]
                mfcc_scaled_line = mfcc_scaled[frame_start:frame_end]

                # length of the padded sequence
                len_2_pad = int(len_seq * np.ceil(len(mfcc_scaled_line) / float(len_seq)))
                len_padded = len_2_pad - len(mfcc_scaled_line)

                # pad feature, label and sample weights
                mfcc_line_pad = np.zeros((len_2_pad, mfcc_scaled_line.shape[1]), dtype='float32')
                mfcc_line_pad[:mfcc_scaled_line.shape[0], :] = mfcc_scaled_line
                mfcc_line_pad = featureReshape(mfcc_line_pad, nlen=7)

                iter_time = len(mfcc_line_pad) / len_seq
                obs_i = np.array([])
                for ii in range(len(mfcc_line_pad) / len_seq):

                    # evaluate for each segment
                    mfcc_line_tensor = mfcc_line_pad[ii * len_seq:(ii + 1) * len_seq]
                    mfcc_line_tensor = np.expand_dims(mfcc_line_tensor, axis=0)
                    mfcc_line_tensor = np.expand_dims(mfcc_line_tensor, axis=2)

                    y_pred = model_keras_cnn_0.predict_on_batch(mfcc_line_tensor)

                    # remove the padded samples
                    if ii == iter_time - 1 and len_padded > 0:
                        y_pred = y_pred[:, :len_seq - len_padded, :]

                    if stateful and ii == iter_time - 1:
                        model_keras_cnn_0.reset_states()

                    # reduce the label dimension
                    y_pred = y_pred.reshape((y_pred.shape[1],))

                    obs_i = np.append(obs_i, y_pred)

                # save onset curve
                print('save onset curve ... ...')
                if not exists(obs_path):
                    makedirs(obs_path)
                pickle.dump(obs_i, open(join(obs_path, obs_filename), 'w'))
            else:
                obs_i = pickle.load(open(join(obs_path, obs_filename), 'r'))

            obs_i = np.squeeze(obs_i)
            obs_i = smooth_obs(obs_i)

            # organize score
            print('Calculating: ', rn, ' phrase', str(i_obs))
            print('ODF Methods: ', architecture)

            duration_score = syllable_durations[i_line]
            duration_score = np.array([float(ds) for ds in duration_score if len(ds)])
            duration_score *= (time_line/np.sum(duration_score))

            i_boundary, label = boundary_decoding(decoding_method=decoding_method,
                                                  obs_i=obs_i,
                                                  duration_score=duration_score,
                                                  varin=varin,
                                                  threshold=threshold,
                                                  hopsize_t=hopsize_t,
                                                  viterbiDecoding=viterbiDecoding,
                                                  OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

            time_boundary_start = np.array(i_boundary[:-1])*hopsize_t
            time_boundary_end = np.array(i_boundary[1:])*hopsize_t

            boundary_list = get_boundary_list(lab=lab,
                                              decoding_method=decoding_method,
                                              time_boundary_start=time_boundary_start,
                                              time_boundary_end=time_boundary_end,
                                              pinyins=pinyins,
                                              syllables=syllables,
                                              i_line=i_line)

            filename_syll_lab = join(eval_results_decoding_path,
                                     artist_path, rn + '_' + str(i_line + 1) + '.syll.lab')

            boundaryLabWriter(boundaryList=boundary_list,
                              outputFilename=filename_syll_lab,
                              label=label)

            if varin['plot'] and obs_cal == 'tocal':
                plot_jingju(nested_syllable_lists,
                            i_line,
                            mfcc_line,
                            hopsize_t,
                            obs_i,
                            i_boundary,
                            duration_score)

    return eval_results_decoding_path


def viterbi_subroutine(test_nacta_2017, test_nacta, eval_label, obs_cal):
    """routine for viterbi decoding"""

    list_recall_onset_25, list_precision_onset_25, list_F1_onset_25 = [], [], []
    list_recall_onset_5, list_precision_onset_5, list_F1_onset_5 = [], [], []
    list_recall_25, list_precision_25, list_F1_25 = [], [], []
    list_recall_5, list_precision_5, list_F1_5 = [], [], []
    for ii in range(5):

        if obs_cal == 'tocal':

            stateful = False if varin['overlap'] else True
            input_shape = (1, len_seq, 1, 80, 15)

            # initialize the model
            model_keras_cnn_0 = jan_original(filter_density=1,
                                             dropout=0.5,
                                             input_shape=input_shape,
                                             batchNorm=False,
                                             dense_activation='sigmoid',
                                             channel=1,
                                             stateful=stateful,
                                             training=False,
                                             bidi=varin['bidi'])

            # load the model weights
            model_keras_cnn_0.load_weights(full_path_keras_cnn_0 + str(ii) + '.h5')

            if varin['dataset'] != 'ismir':
                # evaluate nacta 2017 data set
                batch_process_onset_detection(wav_path=nacta2017_wav_path,
                                              textgrid_path=nacta2017_textgrid_path,
                                              score_path=nacta2017_score_unified_path,
                                              test_recordings=test_nacta_2017,
                                              model_keras_cnn_0=model_keras_cnn_0,
                                              cnnModel_name=cnnModel_name + str(ii),
                                              eval_results_path=eval_results_path + str(ii),
                                              scaler=scaler,
                                              obs_cal=obs_cal,
                                              decoding_method='viterbi',
                                              stateful=stateful)

            # evaluate nacta dataset
            eval_results_decoding_path = batch_process_onset_detection(wav_path=nacta_wav_path,
                                                                       textgrid_path=nacta_textgrid_path,
                                                                       score_path=nacta_score_unified_path,
                                                                       test_recordings=test_nacta,
                                                                       model_keras_cnn_0=model_keras_cnn_0,
                                                                       cnnModel_name=cnnModel_name + str(ii),
                                                                       eval_results_path=eval_results_path + str(ii),
                                                                       scaler=scaler,
                                                                       obs_cal=obs_cal,
                                                                       decoding_method='viterbi',
                                                                       stateful=stateful)
        else:
            eval_results_decoding_path = eval_results_path + str(ii)

        precision_onset, recall_onset, F1_onset, \
        precision, recall, F1, \
            = eval_write_2_txt(eval_result_file_name=join(eval_results_decoding_path, 'results.csv'),
                               segSyllable_path=eval_results_decoding_path,
                               label=eval_label,
                               decoding_method='viterbi')

        list_precision_onset_25.append(precision_onset[0])
        list_precision_onset_5.append(precision_onset[1])
        list_recall_onset_25.append(recall_onset[0])
        list_recall_onset_5.append(recall_onset[1])
        list_F1_onset_25.append(F1_onset[0])
        list_F1_onset_5.append(F1_onset[1])
        list_precision_25.append(precision[0])
        list_precision_5.append(precision[1])
        list_recall_25.append(recall[0])
        list_recall_5.append(recall[1])
        list_F1_25.append(F1[0])
        list_F1_5.append(F1[1])

    return list_precision_onset_25, \
           list_recall_onset_25, \
           list_F1_onset_25, \
           list_precision_25, \
           list_recall_25, \
           list_F1_25, \
           list_precision_onset_5, \
           list_recall_onset_5, \
           list_F1_onset_5, \
           list_precision_5, \
           list_recall_5, \
           list_F1_5


def peak_picking_subroutine(test_nacta_2017, test_nacta, th, obs_cal):
    """routine for peak picking decoding"""
    from src.utilFunctions import append_or_write
    import csv

    eval_result_file_name = join(jingju_results_path,
                                 varin['sample_weighting'],
                                 cnnModel_name + '_peakPicking_threshold_results.txt')

    list_recall_onset_25, list_precision_onset_25, list_F1_onset_25 = [], [], []
    list_recall_onset_5, list_precision_onset_5, list_F1_onset_5 = [], [], []
    list_recall_25, list_precision_25, list_F1_25 = [], [], []
    list_recall_5, list_precision_5, list_F1_5 = [], [], []

    for ii in range(5):

        stateful = False if varin['overlap'] else True
        if obs_cal == 'tocal':
            input_shape = (1, len_seq, 1, 80, 15)
            # initialize the model
            model_keras_cnn_0 = jan_original(filter_density=1,
                                             dropout=0.5,
                                             input_shape=input_shape,
                                             batchNorm=False,
                                             dense_activation='sigmoid',
                                             channel=1,
                                             stateful=stateful,
                                             training=False,
                                             bidi=varin['bidi'])

            model_keras_cnn_0.load_weights(full_path_keras_cnn_0 + str(ii) + '.h5')
        else:
            model_keras_cnn_0 = None

        if varin['dataset'] != 'ismir':
            # evaluate nacta 2017 dataset
            batch_process_onset_detection(wav_path=nacta2017_wav_path,
                                          textgrid_path=nacta2017_textgrid_path,
                                          score_path=nacta2017_score_pinyin_path,
                                          test_recordings=test_nacta_2017,
                                          model_keras_cnn_0=model_keras_cnn_0,
                                          cnnModel_name=cnnModel_name + str(ii),
                                          eval_results_path=eval_results_path + str(ii),
                                          scaler=scaler,
                                          threshold=th,
                                          obs_cal=obs_cal,
                                          decoding_method='peakPicking',
                                          stateful=stateful)

        # evaluate nacta dataset
        eval_results_decoding_path = batch_process_onset_detection(wav_path=nacta_wav_path,
                                                                   textgrid_path=nacta_textgrid_path,
                                                                   score_path=nacta_score_pinyin_path,
                                                                   test_recordings=test_nacta,
                                                                   model_keras_cnn_0=model_keras_cnn_0,
                                                                   cnnModel_name=cnnModel_name + str(ii),
                                                                   eval_results_path=eval_results_path + str(ii),
                                                                   scaler=scaler,
                                                                   threshold=th,
                                                                   obs_cal=obs_cal,
                                                                   decoding_method='peakPicking',
                                                                   stateful=stateful)

        append_write = append_or_write(eval_result_file_name)

        with open(eval_result_file_name, append_write) as testfile:
            csv_writer = csv.writer(testfile)
            csv_writer.writerow([th])

        precision_onset, recall_onset, F1_onset, \
        precision, recall, F1, \
            = eval_write_2_txt(eval_result_file_name,
                               eval_results_decoding_path,
                               label=False,
                               decoding_method='peakPicking')

        list_precision_onset_25.append(precision_onset[0])
        list_precision_onset_5.append(precision_onset[1])
        list_recall_onset_25.append(recall_onset[0])
        list_recall_onset_5.append(recall_onset[1])
        list_F1_onset_25.append(F1_onset[0])
        list_F1_onset_5.append(F1_onset[1])
        list_precision_25.append(precision[0])
        list_precision_5.append(precision[1])
        list_recall_25.append(recall[0])
        list_recall_5.append(recall[1])
        list_F1_25.append(F1[0])
        list_F1_5.append(F1[1])

    return list_precision_onset_25, \
           list_recall_onset_25, \
           list_F1_onset_25, \
           list_precision_25, \
           list_recall_25, \
           list_F1_25, \
           list_precision_onset_5, \
           list_recall_onset_5, \
           list_F1_onset_5, \
           list_precision_5, \
           list_recall_5, \
           list_F1_5


def viterbi_label_eval(test_nacta_2017, test_nacta, eval_label, obs_cal):
    """evaluate viterbi decoding results"""

    list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
    list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5 = \
        viterbi_subroutine(test_nacta_2017=test_nacta_2017,
                           test_nacta=test_nacta,
                           eval_label=eval_label,
                           obs_cal=obs_cal)

    postfix_statistic_sig = 'label' if eval_label else 'nolabel'

    pickle.dump(list_F1_onset_25,
                open(join('./statisticalSignificance/data/jingju', varin['sample_weighting'],
                          cnnModel_name + '_' + 'viterbi' + '_' + postfix_statistic_sig + '.pkl'), 'w'))

    writeResults2Txt(join(jingju_results_path,
                          varin['sample_weighting'],
                          cnnModel_name + '_viterbi' + '_' + postfix_statistic_sig + '.txt'),
                     postfix_statistic_sig,
                     'viterbi',
                     list_precision_onset_25,
                     list_recall_onset_25,
                     list_F1_onset_25,
                     list_precision_25,
                     list_recall_25,
                     list_F1_25,
                     list_precision_onset_5,
                     list_recall_onset_5,
                     list_F1_onset_5,
                     list_precision_5,
                     list_recall_5,
                     list_F1_5)


def peak_picking_eval(test_nacta_2017, test_nacta, obs_cal):
    """evaluate the peak picking results"""

    # coarse search
    best_F1_onset_25, best_th = 0, 0
    for th in range(1, 9):
        th *= 0.1

        try:
            _, _, list_F1_onset_25, _, _, _, _, _, _, _, _, _ = peak_picking_subroutine(test_nacta_2017=test_nacta_2017,
                                                                                        test_nacta=test_nacta,
                                                                                        th=th,
                                                                                        obs_cal=obs_cal)

            if np.mean(list_F1_onset_25) > best_F1_onset_25:
                best_th = th
                best_F1_onset_25 = np.mean(list_F1_onset_25)
        except:
            continue

    # finer scan the best threshold
    for th in range(int((best_th - 0.1) * 100), int((best_th + 0.1) * 100)):
        th *= 0.01

        _, _, list_F1_onset_25, _, _, _, _, _, _, _, _, _ = peak_picking_subroutine(test_nacta_2017=test_nacta_2017,
                                                                                    test_nacta=test_nacta,
                                                                                    th=th,
                                                                                    obs_cal=obs_cal)

        if np.mean(list_F1_onset_25) > best_F1_onset_25:
            best_th = th
            best_F1_onset_25 = np.mean(list_F1_onset_25)

    # get the statistics of the best threshold
    list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
    list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5 = \
        peak_picking_subroutine(test_nacta_2017=test_nacta_2017,
                                test_nacta=test_nacta,
                                th=best_th,
                                obs_cal=obs_cal)

    print('best_th', best_th)

    pickle.dump(list_F1_onset_25,
                open(join('./statisticalSignificance/data/jingju',
                          varin['sample_weighting'],
                          cnnModel_name + '_peakPickingMadmom.pkl'), 'w'))

    writeResults2Txt(
        join(jingju_results_path,
             varin['sample_weighting'],
             cnnModel_name + '_peakPickingMadmom' + '.txt'),
        str(best_th),
        'peakPicking',
        list_precision_onset_25,
        list_recall_onset_25,
        list_F1_onset_25,
        list_precision_25,
        list_recall_25,
        list_F1_25,
        list_precision_onset_5,
        list_recall_onset_5,
        list_F1_onset_5,
        list_precision_5,
        list_recall_5,
        list_F1_5)


if __name__ == '__main__':
    test_nacta_2017, test_nacta = getTestRecordingsScoreDurCorrectionArtistAlbumFilter()

    full_path_mfccBands_2D_scaler_onset = scaler_artist_filter_phrase_model_path

    scaler = pickle.load(open(full_path_mfccBands_2D_scaler_onset, 'rb'))

    obs_cal = 'tocal'

    viterbi_label_eval(test_nacta_2017=test_nacta_2017,
                       test_nacta=test_nacta,
                       eval_label=True,
                       obs_cal=obs_cal)

    obs_cal = 'toload'

    viterbi_label_eval(test_nacta_2017=test_nacta_2017,
                       test_nacta=test_nacta,
                       eval_label=False,
                       obs_cal=obs_cal)

    # peak picking evaluation
    peak_picking_eval(test_nacta_2017=test_nacta_2017,
                      test_nacta=test_nacta,
                      obs_cal=obs_cal)