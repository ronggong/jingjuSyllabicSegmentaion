# -*- coding: utf-8 -*-
import pickle
from os.path import isfile

import numpy as np
import pyximport
from keras.models import load_model

pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from src.file_path_jingju_no_rnn import *
from src.labParser import lab2WordList
from src.parameters_jingju import *
from src.scoreParser import csvDurationScoreParser, csvScorePinyinParser
from src.textgridParser import textGrid2WordList, wordListsParseByLines
from src.trainTestSeparation import getTestTrainRecordingsNactaISMIR

# from peakPicking import peakPicking
from madmom.features.onsets import OnsetPeakPickingProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def takeAllPeaks(obs):
    above_th_idx = []
    peak = []
    peak_idx = []
    th = 0.1
    for ii in range(1, len(obs)):
        if obs[ii] >= th and obs[ii-1] < th:
            above_th_idx = []
            above_th_idx.append(ii)
        elif obs[ii] >= th and obs[ii-1] >= th:
            above_th_idx.append(ii)
        elif obs[ii] < th and obs[ii-1] >= th and len(above_th_idx):
            peak.append(np.max(obs[above_th_idx]))
            peak_idx.append(above_th_idx[np.argmax(obs[above_th_idx])])
    return peak_idx, peak

def onsetFunctionAllRecordings(wav_path,
                               textgrid_path,
                               score_path,
                               scaler,
                               test_recordings,
                               model_keras_cnn_0,
                               model_keras_cnn_1,
                               cnnModel_name_0,
                               cnnModel_name_1,
                               eval_results_path,
                               feature_type='mfcc',
                               dmfcc=False,
                               nbf=True,
                               mth='jordi',
                               late_fusion=True,
                               lab=False,
                               threshold_0=0.54,
                               threshold_1=0.54):
    """
    ODF and viterbi decoding
    :param recordings:
    :param textgrid_path:
    :param dataset_path:
    :param feature_type: 'mfcc', 'mfccBands1D' or 'mfccBands2D'
    :param dmfcc: delta for 'mfcc'
    :param nbf: context frames
    :param mth: jordi, jordi_horizontal_timbral, jan, jan_chan3
    :param late_fusion: Bool
    :return:
    """

    list_peak_0 = []
    list_peak_1 = []
    for artist_path, rn in test_recordings:
        # rn = rn.split('.')[0]

        score_file                  = join(score_path, artist_path, rn+'.csv')

        if not isfile(score_file):
            print 'Score not found: ' + score_file
            continue

        if not lab:
            groundtruth_textgrid_file   = join(textgrid_path, artist_path, rn+'.TextGrid')
            wav_file = join(wav_path, artist_path, rn + '.wav')
        else:
            groundtruth_textgrid_file   = join(textgrid_path, artist_path, rn+'.lab')
            wav_file = join(wav_path, artist_path, rn + '.mp3')

        if not lab:
            lineList        = textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
            utteranceList   = textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

            # parse lines of groundtruth
            nestedUtteranceLists, _, _ = wordListsParseByLines(lineList, utteranceList)

            # parse score
            # syllables, pinyins, syllable_durations, bpm = generatePinyin(score_file)
            syllables, pinyins, syllable_durations, bpm = csvScorePinyinParser(score_file)
        else:
            lineList        = [lab2WordList(groundtruth_textgrid_file, label=True)]
            syllables, syllable_durations, bpm = csvDurationScoreParser(score_file)
        """
        # print(pinyins)
        # print(syllable_durations)

        # load audio
        fs = 44100
        mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
        """
        # mfcc_scaled = scaler.transform(mfcc)
        # mfcc_reshaped = featureReshape(mfcc_scaled, nlen=7)


        # print lineList
        i_line = -1
        for i_obs, line in enumerate(lineList):
            if not lab:
                if len(line[2]) == 0:
                    continue

            i_line += 1

            # if i_line is not
            try:
                print(syllable_durations[i_line])
            except:
                continue

            if float(bpm[i_line]) == 0:
                continue
            """
            if not lab:
                frame_start     = int(round(line[0] / hopsize_t))
                frame_end       = int(round(line[1] / hopsize_t))
            else:
                frame_start = int(round(line[0][0] / hopsize_t))
                frame_end = int(round(line[-1][1] /hopsize_t))
            # print(feature.shape)

            mfcc_line          = mfcc[frame_start:frame_end]
            """
            # mfcc_reshaped_line = mfcc_reshaped[frame_start:frame_end]

            # mfcc_reshaped_line = np.expand_dims(mfcc_reshaped_line, axis=1)
            # obs     = getOnsetFunction(observations=mfcc_reshaped_line,
            #                            model=model_keras_cnn_0,
            #                            method=mth)
            # # obs_i   = obs[:,1]
            #
            # obs_i = obs[:,0]
            #
            # # save onset curve
            # print('save onset curve ... ...')
            # if not exists(obs_path):
            #     makedirs(obs_path)
            # pickle.dump(obs_i, open(join(obs_path, obs_filename), 'w'))

            # plt.figure()
            # plt.plot(obs_i)
            # plt.show()

            obs_path_0 = join('./obs', cnnModel_name_0, artist_path)
            obs_path_1 = join('./obs', cnnModel_name_1, artist_path)
            obs_filename = rn + '_' + str(i_line + 1) + '.pkl'

            obs_i_0 = pickle.load(open(join(obs_path_0, obs_filename), 'r'))
            obs_i_0 = np.squeeze(obs_i_0)
            obs_i_1 = pickle.load(open(join(obs_path_1, obs_filename), 'r'))
            obs_i_1 = np.squeeze(obs_i_1)

            # print(obs_i_0.shape)
            # organize score
            print('Calculating: '+rn+' phrase '+str(i_obs))
            print('ODF Methods: '+mth_ODF+' Late fusion: '+str(fusion))

            if not lab:
                time_line      = line[1] - line[0]
                lyrics_line    = line[2]
                print('Syllable:')
                print(lyrics_line)
            else:
                time_line      = line[-1][1] - line[0][0]

            """
            duration_score = syllable_durations[i_line]
            duration_score = np.array([float(ds) for ds in duration_score if len(ds)])
            duration_score = duration_score * (time_line/np.sum(duration_score))

            # print(duration_score)
            """

            hann = np.hanning(5)
            hann /= np.sum(hann)

            obs_i_0 = np.convolve(hann, obs_i_0, mode='same')
            obs_i_1 = np.convolve(hann, obs_i_1, mode='same')


            if varin['decoding'] == 'viterbi':
                # segmental decoding
                # obs_i_0[0] = 1.0
                # obs_i_0[-1] = 1.0
                # i_boundary_0 = viterbiDecoding.viterbiSegmental2(obs_i_0, duration_score, varin)
                # obs_i_1[0] = 1.0
                # obs_i_1[-1] = 1.0
                # i_boundary_1 = viterbiDecoding.viterbiSegmental2(obs_i_1, duration_score, varin)
                # label = True

                peak_idx_0, peak_0 = takeAllPeaks(obs_i_0)
                peak_idx_1, peak_1 = takeAllPeaks(obs_i_1)

                i_boundary_0 = peak_idx_0 + [len(obs_i_0)]
                i_boundary_1 = peak_idx_1 + [len(obs_i_1)]

                list_peak_0 += peak_0
                list_peak_1 += peak_1
            else:
                # i_boundary = peakPicking(obs_i)
                arg_pp = {'threshold': threshold_0,'smooth':0,'fps': 1./hopsize_t,'pre_max': hopsize_t,'post_max': hopsize_t}
                # peak_picking = OnsetPeakPickingProcessor(threshold=threshold,smooth=smooth,fps=fps,pre_max=pre_max,post_max=post_max)
                peak_picking = OnsetPeakPickingProcessor(**arg_pp)
                i_boundary_0 = peak_picking.process(obs_i_0)
                i_boundary_0 = np.append(i_boundary_0, (len(obs_i_0)-1)*hopsize_t )
                i_boundary_0 /=hopsize_t

                arg_pp = {'threshold': threshold_1, 'smooth': 0, 'fps': 1. / hopsize_t, 'pre_max': hopsize_t,
                          'post_max': hopsize_t}
                # peak_picking = OnsetPeakPickingProcessor(threshold=threshold,smooth=smooth,fps=fps,pre_max=pre_max,post_max=post_max)
                peak_picking = OnsetPeakPickingProcessor(**arg_pp)
                i_boundary_1 = peak_picking.process(obs_i_1)
                i_boundary_1 = np.append(i_boundary_1, (len(obs_i_1) - 1) * hopsize_t)
                i_boundary_1 /= hopsize_t
                label = False


            # print(np.array(groundtruth_syllable)*fs/hopsize)

            if varin['plot']:
                print(lineList)
                nestedUL = nestedUtteranceLists[i_line][1]
                print(nestedUL)
                groundtruth_onset = [l[0]-nestedUL[0][0] for l in nestedUL]

                # nestedUL = lineList[0]
                # groundtruth_onset = [l[0]-line[0] for l in nestedUL]
                # groundtruth_syllables = [l[2] for l in nestedUL]

                # plot Error analysis figures
                plt.figure(figsize=(16, 8))
                # plt.figure(figsize=(8, 4))
                # class weight
                ax1 = plt.subplot(411)
                y = np.arange(0, 80)
                x = np.arange(0, mfcc_line.shape[0])*hopsize_t
                cax = plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 10:80 * 11]))
                for i_gs, gs in enumerate(groundtruth_onset):
                    plt.axvline(gs, color='r', linewidth=2)
                    # plt.text(gs, ax1.get_ylim()[1], groundtruth_syllables[i_gs])

                # cbar = fig.colorbar(cax)
                ax1.set_ylabel('Mel bands', fontsize=12)
                ax1.get_xaxis().set_visible(False)
                ax1.axis('tight')
                # plt.title('Calculating: '+rn+' phrase '+str(i_obs))

                ax2 = plt.subplot(412, sharex=ax1)
                plt.plot(np.arange(0,len(obs_i_0))*hopsize_t, obs_i_0)
                for i_ib in range(len(i_boundary_0)-1):
                    plt.axvline(i_boundary_0[i_ib] * hopsize_t, color='r', linewidth=2)
                    # plt.text(i_boundary[i_ib] * hopsize_t, ax2.get_ylim()[1], syllables[i_line][i_ib])

                ax2.set_ylabel('ODF', fontsize=12)
                ax2.axis('tight')

                ax2 = plt.subplot(413, sharex=ax1)
                plt.plot(np.arange(0, len(obs_i_1)) * hopsize_t, obs_i_1)
                for i_ib in range(len(i_boundary_1) - 1):
                    plt.axvline(i_boundary_1[i_ib] * hopsize_t, color='r', linewidth=2)
                    # plt.text(i_boundary[i_ib] * hopsize_t, ax2.get_ylim()[1], syllables[i_line][i_ib])

                ax2.set_ylabel('ODF', fontsize=12)
                ax2.axis('tight')


                ax4 = plt.subplot(414, sharex=ax1)
                print(duration_score)
                time_start = 0
                for ii_ds, ds in enumerate(duration_score):
                    ax4.add_patch(
                        patches.Rectangle(
                            (time_start, ii_ds),  # (x,y)
                            ds,  # width
                            1,  # height
                        ))
                    time_start += ds
                ax4.set_ylim((0,len(duration_score)))
                ax4.set_ylabel('Score duration', fontsize=12)
                plt.xlabel('Time (s)')
                # plt.tight_layout()

                plt.show()

    return peak_0, peak_1


if __name__ == '__main__':

    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsNactaISMIR()

    # load scaler
    filename_scaler_onset = 'scaler_syllable_mfccBands2D_old+new_ismir_madmom.pkl'
    full_path_mfccBands_2D_scaler_onset = join(cnnModels_path, filename_scaler_onset)
    scaler = pickle.load(open(full_path_mfccBands_2D_scaler_onset, 'rb'))

    if varin['decoding'] == 'viterbi':
        list_peak_0, list_peak_1 = [], []
        for ii in range(5):
            # load model 0
            filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_ismir_madmom_early_stopping'
            cnnModel_name_0 = 'jan_old+new_ismir_madmom_early_stopping'
            full_path_keras_cnn_0 = join(cnnModels_path, filename_keras_cnn_0)
            model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(ii) + '.h5')

            filename_keras_cnn_1 = 'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_ismir_madmom_early_stopping'
            cnnModel_name_1 = 'jordi_temporal_ismir_madmom_early_stopping'
            full_path_keras_cnn_1 = join(cnnModels_path, filename_keras_cnn_1)
            model_keras_cnn_1 = load_model(full_path_keras_cnn_1 + str(ii) + '.h5')

            # nacta2017
            # onsetFunctionAllRecordings(wav_path=nacta2017_wav_path,
            #                            textgrid_path=nacta2017_textgrid_path,
            #                            score_path=nacta2017_score_pinyin_path,
            #                            test_recordings=testNacta2017,
            #                            feature_type='mfccBands2D',
            #                            dmfcc=False,
            #                            nbf=True,
            #                            mth=mth_ODF,
            #                            late_fusion=fusion)

            # nacta
            peak_0, peak_1 = eval_results_decoding_path = onsetFunctionAllRecordings(wav_path=nacta_wav_path,
                                                                   textgrid_path=nacta_textgrid_path,
                                                                   score_path=nacta_score_pinyin_path,
                                                                   test_recordings=testNacta,
                                                                   model_keras_cnn_0=model_keras_cnn_0,
                                                                    model_keras_cnn_1=model_keras_cnn_1,
                                                                   cnnModel_name_0=cnnModel_name_0+str(ii),
                                                                    cnnModel_name_1=cnnModel_name_1+str(ii),
                                                                    eval_results_path=eval_results_path+str(ii),
                                                                   scaler=scaler,
                                                                   feature_type='madmom',
                                                                   dmfcc=False,
                                                                   nbf=True,
                                                                   mth=mth_ODF,
                                                                   late_fusion=fusion)
            list_peak_0 += peak_0
            list_peak_1 += peak_1

        print(np.mean(list_peak_0), np.std(list_peak_0))
        print(np.mean(list_peak_1), np.std(list_peak_1))


    else:
        for ii in range(5):
            filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_ismir_madmom_early_stopping'
            cnnModel_name_0 = 'jan_old+new_ismir_madmom_early_stopping'
            full_path_keras_cnn_0 = join(cnnModels_path, filename_keras_cnn_0)
            model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(ii) + '.h5')

            filename_keras_cnn_1 = 'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_ismir_madmom_early_stopping'
            cnnModel_name_1 = 'jordi_temporal_ismir_madmom_early_stopping'
            full_path_keras_cnn_1 = join(cnnModels_path, filename_keras_cnn_1)
            model_keras_cnn_1 = load_model(full_path_keras_cnn_1 + str(ii) + '.h5')

            eval_results_decoding_path = onsetFunctionAllRecordings(wav_path=nacta_wav_path,
                                                                    textgrid_path=nacta_textgrid_path,
                                                                    score_path=nacta_score_pinyin_path,
                                                                    test_recordings=testNacta,
                                                                    model_keras_cnn_0=model_keras_cnn_0,
                                                                    model_keras_cnn_1=model_keras_cnn_1,
                                                                    cnnModel_name_0=cnnModel_name_0 + str(ii),
                                                                    cnnModel_name_1=cnnModel_name_1 + str(ii),
                                                                    eval_results_path=eval_results_path + str(ii),
                                                                    scaler=scaler,
                                                                    feature_type='madmom',
                                                                    dmfcc=False,
                                                                    nbf=True,
                                                                    mth=mth_ODF,
                                                                    late_fusion=fusion,
                                                                    threshold_0=0.52,
                                                                    threshold_1=0.39)

