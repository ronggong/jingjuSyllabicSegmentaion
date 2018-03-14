# -*- coding: utf-8 -*-
import pickle
from os import makedirs
from os.path import isfile, exists

import numpy as np
import pyximport

pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from src.file_path_jingju_rnn import *
from src.labWriter import boundaryLabWriter
from src.labParser import lab2WordList
# from src.parameters import *
from src.scoreParser import csvDurationScoreParser, csvScorePinyinParser
from src.textgridParser import textGrid2WordList, wordListsParseByLines
from src.utilFunctions import featureReshape
from src.trainTestSeparation import getTestTrainRecordingsNactaISMIR, \
    getTestRecordingsScoreDurCorrectionArtistAlbumFilter

# from peakPicking import peakPicking
from madmom.features.onsets import OnsetPeakPickingProcessor
from datasetCollection.trainingSampleCollection import getMFCCBands2DMadmom
from training_scripts.models_CRNN import jan_original
import viterbiDecoding
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# cnnModel_name = 'jordi_temporal_ismir_madmom'
# cnnModel_name_2 = 'jordi_timbral_ismir_madmom'

# print(full_path_keras_cnn_0)
# model_keras_cnn_0 = load_model(full_path_keras_cnn_0)
# model_keras_cnn_1 = load_model(full_path_keras_cnn_1)


def onsetFunctionAllRecordings(wav_path,
                               textgrid_path,
                               score_path,
                               scaler,
                               test_recordings,
                               model_keras_cnn_0,
                                cnnModel_name,
                               eval_results_path,
                               feature_type='mfcc',
                               dmfcc=False,
                               nbf=True,
                               mth='jordi',
                               late_fusion=True,
                               lab=False,
                               threshold=0.54,
                               obs_cal=True,
                               decoding_method='viterbi',
                               stateful=True):
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


    for artist_path, rn in test_recordings:
        # rn = rn.split('.')[0]

        score_file                  = join(score_path, artist_path, rn+'.csv')
        print(score_file)

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

        # print(pinyins)
        # print(syllable_durations)

        if obs_cal == 'tocal':
            # load audio
            fs = 44100
            mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
            mfcc_scaled = scaler.transform(mfcc)


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

            if not lab:
                frame_start     = int(round(line[0] / hopsize_t))
                frame_end       = int(round(line[1] / hopsize_t))
            else:
                frame_start = int(round(line[0][0] / hopsize_t))
                frame_end = int(round(line[-1][1] /hopsize_t))
            # print(feature.shape)

            obs_path = join('./obs', cnnModel_name, artist_path)
            obs_filename = rn + '_' + str(i_line + 1) + '.pkl'

            if obs_cal == 'tocal':
                mfcc_line          = mfcc[frame_start:frame_end]
                mfcc_scaled_line = mfcc_scaled[frame_start:frame_end]

                # length of the paded sequence
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

                # plt.figure()
                # plt.plot(obs_i)
                # plt.show()
            else:
                obs_i = pickle.load(open(join(obs_path, obs_filename), 'r'))

            obs_i = np.squeeze(obs_i)

            print(obs_i.shape)
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


            duration_score = syllable_durations[i_line]
            duration_score = np.array([float(ds) for ds in duration_score if len(ds)])
            duration_score = duration_score * (time_line/np.sum(duration_score))

            # print(duration_score)

            hann = np.hanning(5)
            hann /= np.sum(hann)

            obs_i = np.convolve(hann, obs_i, mode='same')

            if decoding_method == 'viterbi':
                # segmental decoding
                obs_i[0] = 1.0
                obs_i[-1] = 1.0
                i_boundary = viterbiDecoding.viterbiSegmental2(obs_i, duration_score, varin)

                if varin['corrected_score_duration']:
                    eval_results_decoding_path = eval_results_path + '_corrected_score_duration'
                else:
                    eval_results_decoding_path = eval_results_path

                filename_syll_lab = join(eval_results_decoding_path, artist_path, rn + '_' + str(i_line + 1) + '.syll.lab')
                label = True
            else:
                # i_boundary = peakPicking(obs_i)
                arg_pp = {'threshold': threshold,'smooth':0,'fps': 1./hopsize_t,'pre_max': hopsize_t,'post_max': hopsize_t}
                # peak_picking = OnsetPeakPickingProcessor(threshold=threshold,smooth=smooth,fps=fps,pre_max=pre_max,post_max=post_max)
                peak_picking = OnsetPeakPickingProcessor(**arg_pp)
                i_boundary = peak_picking.process(obs_i)
                i_boundary = np.append(i_boundary, (len(obs_i)-1)*hopsize_t )
                i_boundary /=hopsize_t
                label = False
                eval_results_decoding_path = eval_results_path+'_peakPickingMadmom'
                filename_syll_lab = join(eval_results_decoding_path, artist_path, rn + '_' + str(i_line + 1) + '.syll.lab')

            time_boundray_start = np.array(i_boundary[:-1])*hopsize_t
            time_boundray_end   = np.array(i_boundary[1:])*hopsize_t

            # uncomment this section if we want to write boundaries to .syll.lab file

            eval_results_data_path = dirname(filename_syll_lab)

            print(eval_results_data_path)

            if not exists(eval_results_data_path):
                makedirs(eval_results_data_path)

            # write boundary lab file
            if not lab:
                if decoding_method == 'viterbi':
                    boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist(), filter(None,pinyins[i_line]))
                else:
                    boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist())

            else:
                if decoding_method == 'viterbi':
                    boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist(), syllables[i_line])
                    label = True

                else:
                    boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist())
                    label = False

            boundaryLabWriter(boundaryList=boundary_list,
                              outputFilename=filename_syll_lab,
                                label=label)

            print(i_boundary)
            print(len(obs_i))
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
                plt.figure(figsize=(16, 6))
                # plt.figure(figsize=(8, 4))
                # class weight
                ax1 = plt.subplot(3,1,1)
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

                ax2 = plt.subplot(312, sharex=ax1)
                plt.plot(np.arange(0,len(obs_i))*hopsize_t, obs_i)
                for i_ib in range(len(i_boundary)-1):
                    plt.axvline(i_boundary[i_ib] * hopsize_t, color='r', linewidth=2)
                    # plt.text(i_boundary[i_ib] * hopsize_t, ax2.get_ylim()[1], syllables[i_line][i_ib])

                ax2.set_ylabel('ODF', fontsize=12)
                ax2.axis('tight')


                ax3 = plt.subplot(313, sharex=ax1)
                print(duration_score)
                time_start = 0
                for ii_ds, ds in enumerate(duration_score):
                    ax3.add_patch(
                        patches.Rectangle(
                            (time_start, ii_ds),  # (x,y)
                            ds,  # width
                            1,  # height
                        ))
                    time_start += ds
                ax3.set_ylim((0,len(duration_score)))
                ax3.set_ylabel('Score duration', fontsize=12)
                plt.xlabel('Time (s)')
                # plt.tight_layout()

                plt.show()
    return eval_results_decoding_path


if __name__ == '__main__':

    from eval_demo import eval_write_2_txt

    if varin['dataset'] == 'ismir':
        testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsNactaISMIR()
        full_path_mfccBands_2D_scaler_onset = scaler_ismir_phrase_model_path
    else:
        testNacta2017, testNacta = getTestRecordingsScoreDurCorrectionArtistAlbumFilter()
        full_path_mfccBands_2D_scaler_onset = scaler_artist_filter_phrase_model_path


    scaler = pickle.load(open(full_path_mfccBands_2D_scaler_onset, 'rb'))

    def peakPickingSubroutine(th, obs_cal):
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
            if obs_cal=='tocal':
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
                # nacta2017
                onsetFunctionAllRecordings(wav_path=nacta2017_wav_path,
                                           textgrid_path=nacta2017_textgrid_path,
                                           score_path=nacta2017_score_pinyin_path,
                                           test_recordings=testNacta2017,
                                           model_keras_cnn_0=model_keras_cnn_0,
                                           cnnModel_name=cnnModel_name + str(ii),
                                           eval_results_path=eval_results_path + str(ii),
                                           scaler=scaler,
                                           feature_type='madmom',
                                           dmfcc=False,
                                           nbf=True,
                                           mth=mth_ODF,
                                           late_fusion=fusion,
                                           threshold=th,
                                           obs_cal=obs_cal,
                                           decoding_method='peakPicking',
                                           stateful=stateful)

            eval_results_decoding_path = onsetFunctionAllRecordings(wav_path=nacta_wav_path,
                                                                    textgrid_path=nacta_textgrid_path,
                                                                    score_path=nacta_score_pinyin_path,
                                                                    test_recordings=testNacta,
                                                                    model_keras_cnn_0=model_keras_cnn_0,
                                                                    cnnModel_name=cnnModel_name + str(ii),
                                                                    eval_results_path=eval_results_path + str(ii),
                                                                    scaler=scaler,
                                                                    feature_type='madmom',
                                                                    dmfcc=False,
                                                                    nbf=True,
                                                                    mth=mth_ODF,
                                                                    late_fusion=fusion,
                                                                    threshold=th,
                                                                    obs_cal=obs_cal,
                                                                    decoding_method='peakPicking',
                                                                    stateful=stateful)

            append_write = append_or_write(eval_result_file_name)
            with open(eval_result_file_name, append_write) as testfile:
                csv_writer = csv.writer(testfile)
                csv_writer.writerow([th])

            # eval_results_decoding_path = cnnModel_name + str(ii) + '_peakPickingMadmom'
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

        return list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
               list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5


    def viterbiSubroutine(eval_label, obs_cal):

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

                model_keras_cnn_0.load_weights(full_path_keras_cnn_0 + str(ii) + '.h5')

                if varin['dataset'] != 'ismir':
                    # nacta2017
                    onsetFunctionAllRecordings(wav_path=nacta2017_wav_path,
                                               textgrid_path=nacta2017_textgrid_path,
                                               score_path=nacta2017_score_unified_path,
                                               test_recordings=testNacta2017,
                                               model_keras_cnn_0=model_keras_cnn_0,
                                               cnnModel_name=cnnModel_name + str(ii),
                                               eval_results_path=eval_results_path + str(ii),
                                               scaler=scaler,
                                               feature_type='madmom',
                                               dmfcc=False,
                                               nbf=True,
                                               mth=mth_ODF,
                                               late_fusion=fusion,
                                               obs_cal=obs_cal,
                                               decoding_method='viterbi',
                                               stateful=stateful)

                # nacta
                eval_results_decoding_path = onsetFunctionAllRecordings(wav_path=nacta_wav_path,
                                                                        textgrid_path=nacta_textgrid_path,
                                                                        score_path=nacta_score_unified_path,
                                                                        test_recordings=testNacta,
                                                                        model_keras_cnn_0=model_keras_cnn_0,
                                                                        cnnModel_name=cnnModel_name + str(ii),
                                                                        eval_results_path=eval_results_path + str(ii),
                                                                        scaler=scaler,
                                                                        feature_type='madmom',
                                                                        dmfcc=False,
                                                                        nbf=True,
                                                                        mth=mth_ODF,
                                                                        late_fusion=fusion,
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

        return list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
               list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5


    def writeResults2Txt(filename,
                         eval_label_str,
                         decoding_method,
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
                         list_F1_5):
        """
        :param filename:
        :param eval_label_str: eval label or not
        :param decoding_method: viterbi or peakPicking
        :param list_precision_onset_25:
        :param list_recall_onset_25:
        :param list_F1_onset_25:
        :param list_precision_25:
        :param list_recall_25:
        :param list_F1_25:
        :param list_precision_onset_5:
        :param list_recall_onset_5:
        :param list_F1_onset_5:
        :param list_precision_5:
        :param list_recall_5:
        :param list_F1_5:
        :return:
        """

        with open(filename, 'w') as f:
            f.write(decoding_method)
            f.write('\n')
            f.write(eval_label_str)
            f.write('\n')
            f.write(str(np.mean(list_precision_onset_25))+' '+str(np.std(list_precision_onset_25)))
            f.write('\n')
            f.write(str(np.mean(list_recall_onset_25))+' '+str(np.std(list_recall_onset_25)))
            f.write('\n')
            f.write(str(np.mean(list_F1_onset_25))+' '+str(np.std(list_F1_onset_25)))
            f.write('\n')

            f.write(str(np.mean(list_precision_25))+' '+str(np.std(list_precision_25)))
            f.write('\n')
            f.write(str(np.mean(list_recall_25))+' '+str(np.std(list_recall_25)))
            f.write('\n')
            f.write(str(np.mean(list_F1_25))+' '+str(np.std(list_F1_25)))
            f.write('\n')

            f.write(str(np.mean(list_precision_onset_5)) + ' ' + str(np.std(list_precision_onset_5)))
            f.write('\n')
            f.write(str(np.mean(list_recall_onset_5)) + ' ' + str(np.std(list_recall_onset_5)))
            f.write('\n')
            f.write(str(np.mean(list_F1_onset_5)) + ' ' + str(np.std(list_F1_onset_5)))
            f.write('\n')

            f.write(str(np.mean(list_precision_5)) + ' ' + str(np.std(list_precision_5)))
            f.write('\n')
            f.write(str(np.mean(list_recall_5)) + ' ' + str(np.std(list_recall_5)))
            f.write('\n')
            f.write(str(np.mean(list_F1_5)) + ' ' + str(np.std(list_F1_5)))


    def viterbiLabelEval(eval_label, obs_cal):

        list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
        list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5 = \
            viterbiSubroutine(eval_label, obs_cal)

        postfix_statistic_sig = 'label' if eval_label else 'nolabel'
        pickle.dump(list_F1_onset_25,
                    open(join('./statisticalSignificance/data/jingju', varin['sample_weighting'],
                              cnnModel_name + '_' + 'viterbi' + '_' + postfix_statistic_sig + '.pkl'), 'w'))

        writeResults2Txt(join(jingju_results_path, varin['sample_weighting'], cnnModel_name + '_viterbi' + '_' + postfix_statistic_sig + '.txt'),
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

    ##-- viterbi evaluation

    obs_cal = 'tocal'

    viterbiLabelEval(eval_label=True, obs_cal=obs_cal)

    obs_cal = 'toload'

    viterbiLabelEval(eval_label=False, obs_cal=obs_cal)

    ##-- peak picking evaluation
    # scan the best threshold
    best_F1_onset_25, best_th = 0, 0
    for th in range(1, 9):
        th *= 0.1

        try:
            _,_,list_F1_onset_25,_,_,_,_,_,_,_,_,_= peakPickingSubroutine(th, obs_cal)

            if np.mean(list_F1_onset_25) > best_F1_onset_25:
                best_th = th
                best_F1_onset_25 = np.mean(list_F1_onset_25)
        except:
            continue

    # finer scan the best threshold
    for th in range(int((best_th - 0.1) * 100), int((best_th + 0.1) * 100)):
        th *= 0.01

        _,_,list_F1_onset_25,_,_,_,_,_,_,_,_,_= peakPickingSubroutine(th, obs_cal)

        if np.mean(list_F1_onset_25) > best_F1_onset_25:
            best_th = th
            best_F1_onset_25 = np.mean(list_F1_onset_25)

    # get the statistics of the best th
    list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
    list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5 = \
        peakPickingSubroutine(best_th, obs_cal)

    print('best_th', best_th)

    pickle.dump(list_F1_onset_25,
                open(join('./statisticalSignificance/data/jingju', varin['sample_weighting'], cnnModel_name + '_peakPickingMadmom.pkl'), 'w'))

    writeResults2Txt(join(jingju_results_path, varin['sample_weighting'], cnnModel_name + '_peakPickingMadmom' + '.txt'),
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

    # # Riyaz
    # testRiyaz, trainRiyaz = getTestTrainrecordingsRiyaz()
    #
    # onsetFunctionAllRecordings(wav_path=riyaz_mp3_path,
    #                            textgrid_path=riyaz_groundtruthlab_path,
    #                            score_path=riyaz_score_path,
    #                            test_recordings=testRiyaz,
    #                            feature_type='mfccBands2D',
    #                            dmfcc=False,
    #                            nbf=True,
    #                            mth=mth_ODF,
    #                            late_fusion=fusion,
    #                            lab=True)