# -*- coding: utf-8 -*-
import pickle
from os import makedirs, walk, listdir
from os.path import isfile, exists, dirname, join

import essentia.standard as ess
import numpy as np
from keras.models import load_model

import pyximport
pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from src.filePath import *
from src.labWriter import boundaryLabWriter
from src.labParser import lab2WordList
from src.parameters import *
from src.scoreManip import phonemeDurationForLine
from src.scoreParser import generatePinyin, csvDurationScoreParser, csvScorePinyinParser
from src.textgridParser import textGrid2WordList, wordListsParseByLines
from trainingSampleCollection import featureReshape
from trainingSampleCollection import getMFCCBands2D
from src.trainTestSeparation import getTestTrainRecordingsMaleFemale, \
    getTestTrainrecordingsRiyaz, \
    getTestTrainRecordingsNactaISMIR, \
    getTestTrainRecordingsArtistAlbumFilter, \
    getTestTrainRecordingsArtist

from peakPicking import peakPicking
import viterbiDecoding
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def getOnsetFunction(observations, path_keras_cnn, method='jan'):
    """
    Load CNN model to calculate ODF
    :param observations:
    :return:
    """

    model = load_model(path_keras_cnn)
    print(path_keras_cnn)

    ##-- call pdnn to calculate the observation from the features
    # if method=='jordi':
    #     observations = [observations, observations, observations, observations, observations, observations]
    # elif method=='jordi_horizontal_timbral':
    #     observations = [observations, observations, observations, observations, observations, observations,
    #                     observations, observations, observations, observations, observations, observations]
    if method == 'jordi':
        obs = model.predict(observations, batch_size=128, verbose=1)
    else:
        obs = model.predict_proba(observations, batch_size=128)
    return obs


def featureExtraction(audio_monoloader, scaler, framesize_t, hopsize_t, fs, dmfcc=False, nbf=False, feature_type='mfccBands2D'):
    """
    extract mfcc features
    :param audio_monoloader:
    :param scaler:
    :param dmfcc:
    :param nbf:
    :param feature_type:
    :return:
    """
    if feature_type == 'mfccBands2D':
        mfcc = getMFCCBands2D(audio_monoloader, framesize_t, hopsize_t, fs, nbf=nbf, nlen=varin['nlen'])
        mfcc = np.log(100000 * mfcc + 1)

        mfcc = np.array(mfcc, dtype='float32')
        mfcc_scaled = scaler.transform(mfcc)
        mfcc_reshaped = featureReshape(mfcc_scaled)
    else:
        print(feature_type + ' is not exist.')
        raise
    return mfcc, mfcc_reshaped


def trackOnsetPosByPath(path, idx_syllable_start_state):
    idx_onset = []
    for ii in range(len(path)-1):
        if path[ii+1] != path[ii] and path[ii+1] in idx_syllable_start_state:
            idx_onset.append(ii)
    return idx_onset

def late_fusion_calc(obs_0, obs_1, mth=0, coef=0.5):
    """
    Late fusion methods
    :param obs_0:
    :param obs_1:
    :param mth: 0-addition 1-addition with linear norm 2-exponential weighting mulitplication with linear norm
    3-multiplication 4- multiplication with linear norm
    :param coef: weighting coef for multiplication
    :return:
    """
    assert len(obs_0) == len(obs_1)

    obs_out = []

    if mth==1 or mth==2 or mth==4:
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        obs_0 = obs_0.reshape((len(obs_0),1))
        obs_1 = obs_1.reshape((len(obs_1),1))
        # print(obs_0.shape, obs_1.shape)
        obs_0 = min_max_scaler.fit_transform(obs_0)
        obs_1 = min_max_scaler.fit_transform(obs_1)

    if mth == 0 or mth == 1:
        # addition
        obs_out = np.add(obs_0, obs_1)/2
    elif mth == 2:
        # multiplication with exponential weighting
        obs_out = np.multiply(np.power(obs_0, coef), np.power(obs_1, 1-coef))
    elif mth == 3 or mth == 4:
        # multiplication
        obs_out = np.multiply(obs_0, obs_1)

    return obs_out

def late_fusion_calc_3(obs_0, obs_1, obs_2, mth=2, coef=0.33333333):
    """
    Late fusion methods
    :param obs_0:
    :param obs_1:
    :param mth: 0-addition 1-addition with linear norm 2-exponential weighting mulitplication with linear norm
    3-multiplication 4- multiplication with linear norm
    :param coef: weighting coef for multiplication
    :return:
    """
    assert len(obs_0) == len(obs_1)

    obs_out = []

    if mth==2:
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        obs_0 = obs_0.reshape((len(obs_0),1))
        obs_1 = obs_1.reshape((len(obs_1),1))
        obs_2 = obs_2.reshape((len(obs_2),1))

        # print(obs_0.shape, obs_1.shape)
        obs_0 = min_max_scaler.fit_transform(obs_0)
        obs_1 = min_max_scaler.fit_transform(obs_1)
        obs_2 = min_max_scaler.fit_transform(obs_2)


    if mth == 2:
        # multiplication with exponential weighting
        obs_out = np.multiply(np.power(obs_0, coef), np.power(obs_1, coef))
        obs_out = np.multiply(obs_out, np.power(obs_2, 1-coef))
    else:
        raise ValueError

    return obs_out


def onsetFunctionAllRecordings(wav_path,
                               textgrid_path,
                               score_path,
                               test_recordings,
                               feature_type='mfcc',
                               dmfcc=False,
                               nbf=True,
                               mth='jordi',
                               late_fusion=True,
                               lab=False):
    """
    ODF and viter decoding
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

    scaler = pickle.load(open(full_path_mfccBands_2D_scaler_onset, 'rb'))
    if mth == 'jan_chan3':
        scaler_23 = pickle.load(open(full_path_mfccBands_2D_scaler_onset_23, 'rb'))
        scaler_46 = pickle.load(open(full_path_mfccBands_2D_scaler_onset_46, 'rb'))
        scaler_93 = pickle.load(open(full_path_mfccBands_2D_scaler_onset_93, 'rb'))

    # kerasModel = _LRHMM.kerasModel(full_path_keras_cnn_am)

    # list_file_path_name = []
    # for file_path_name in walk(score_path):
    #     list_file_path_name.append(file_path_name)
    #
    # list_artist_level_path = list_file_path_name[0][1]
    #
    #
    # for artist_path in list_artist_level_path:
    #     textgrid_artist_path = join(textgrid_path, artist_path)
    #     recording_names = [f for f in listdir(textgrid_artist_path) if isfile(join(textgrid_artist_path, f))]
    #     print(recording_names)


    for artist_path, rn in test_recordings:
        # rn = rn.split('.')[0]

        if not lab:
            groundtruth_textgrid_file   = join(textgrid_path, artist_path, rn+'.TextGrid')
            wav_file = join(wav_path, artist_path, rn + '.wav')
        else:
            groundtruth_textgrid_file   = join(textgrid_path, artist_path, rn+'.lab')
            wav_file = join(wav_path, artist_path, rn + '.mp3')

        score_file                  = join(score_path, artist_path, rn+'.csv')

        if not isfile(score_file):
            print 'Score not found: ' + score_file
            continue

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

        # load audio
        fs = 44100
        if not lab:
            audio_monoloader               = ess.MonoLoader(downmix = 'left', filename = wav_file, sampleRate = fs)()
        else:
            audio, fs, nc, md5, br, codec = ess.AudioLoader(filename=wav_file)()
            audio_monoloader = audio[:, 0]  # take the left channel

        if mth == 'jordi' or mth == 'jordi_horizontal_timbral' or mth == 'jan':
            mfcc, mfcc_reshaped = featureExtraction(audio_monoloader,
                                                          scaler,
                                                          framesize_t,
                                                            hopsize_t,
                                                            fs,
                                                          dmfcc=dmfcc,
                                                          nbf=nbf,
                                                          feature_type='mfccBands2D')
        elif mth == 'jan_chan3':
            # for jan 3 channels input
            mfcc_23, mfcc_reshaped_23 = featureExtraction(audio_monoloader,
                                                    scaler_23,
                                                          framesize_t,
                                                          hopsize_t,
                                                          fs,
                                                    dmfcc=dmfcc,
                                                    nbf=nbf,
                                                    feature_type='mfccBands2D')

            mfcc_46, mfcc_reshaped_46 = featureExtraction(audio_monoloader,
                                                    scaler_46,
                                                          framesize_t,
                                                          hopsize_t,
                                                          fs,
                                                    dmfcc=dmfcc,
                                                    nbf=nbf,
                                                    feature_type='mfccBands2D')

            mfcc_93, mfcc_reshaped_93 = featureExtraction(audio_monoloader,
                                                    scaler_93,
                                                          framesize_t,
                                                          hopsize_t,
                                                          fs,
                                                    dmfcc=dmfcc,
                                                    nbf=nbf,
                                                    feature_type='mfccBands2D')

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

            if mth == 'jordi' or mth == 'jordi_horizontal_timbral' or mth == 'jan':
                mfcc_line          = mfcc[frame_start:frame_end]
                mfcc_reshaped_line = mfcc_reshaped[frame_start:frame_end]
            elif mth == 'jan_chan3':
                mfcc_line_23 = mfcc_23[frame_start:frame_end]
                mfcc_reshaped_line_23 = mfcc_reshaped_23[frame_start:frame_end]
                mfcc_reshaped_line_23 = mfcc_reshaped_line_23[...,np.newaxis]

                mfcc_line_46 = mfcc_46[frame_start:frame_end]
                mfcc_reshaped_line_46 = mfcc_reshaped_46[frame_start:frame_end]
                mfcc_reshaped_line_46 = mfcc_reshaped_line_46[...,np.newaxis]

                mfcc_line_93 = mfcc_93[frame_start:frame_end]
                mfcc_reshaped_line_93 = mfcc_reshaped_93[frame_start:frame_end]
                mfcc_reshaped_line_93 = mfcc_reshaped_line_93[...,np.newaxis]

                mfcc_reshaped_line = np.concatenate((mfcc_reshaped_line_23,mfcc_reshaped_line_46,mfcc_reshaped_line_93),axis=3)

            mfcc_reshaped_line = np.expand_dims(mfcc_reshaped_line, axis=1)
            obs     = getOnsetFunction(observations=mfcc_reshaped_line,
                                       path_keras_cnn=full_path_keras_cnn_0,
                                       method=mth)
            # obs_i   = obs[:,1]

            obs_i = obs[:,0]


            hann = np.hanning(5)
            hann /= np.sum(hann)

            obs_i = np.convolve(hann, obs_i, mode='same')

            # plt.figure()
            # plt.plot(obs_i)
            # plt.show()

            if late_fusion:
                # fuse second observation
                obs_2 = getOnsetFunction(observations=mfcc_reshaped_line,
                                         path_keras_cnn=full_path_keras_cnn_1,
                                         method=mth)
                obs_2_i = obs_2[:, 0]
                obs_2_i = np.convolve(hann, obs_2_i, mode='same')

                # fuse the third observation
                obs_3 = getOnsetFunction(observations=mfcc_reshaped_line,
                                         path_keras_cnn=full_path_keras_cnn_2,
                                         method=mth)
                obs_3_i = obs_3[:, 0]
                obs_3_i = np.convolve(hann, obs_3_i, mode='same')
                obs_i = late_fusion_calc_3(obs_i, obs_2_i, obs_3_i, mth=2)

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

            if varin['decoding'] == 'viterbi':
                # segmental decoding
                obs_i[0] = 1.0
                obs_i[-1] = 1.0
                i_boundary = viterbiDecoding.viterbiSegmental2(obs_i, duration_score, varin)
                filename_syll_lab = join(eval_results_path, artist_path, rn + '_' + str(i_line + 1) + '.syll.lab')
                label = True
            else:
                i_boundary = peakPicking(obs_i)
                label = False
                filename_syll_lab = join(eval_results_path+'_peakPicking', artist_path, rn + '_' + str(i_line + 1) + '.syll.lab')

            time_boundray_start = np.array(i_boundary[:-1])*hopsize_t
            time_boundray_end   = np.array(i_boundary[1:])*hopsize_t

            # uncomment this section if we want to write boundaries to .syll.lab file

            eval_results_data_path = dirname(filename_syll_lab)

            if not exists(eval_results_data_path):
                makedirs(eval_results_data_path)

            # write boundary lab file
            if not lab:
                boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist(), filter(None,pinyins[i_line]))
            else:
                boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist(), syllables[i_line])
                label = True

            boundaryLabWriter(boundaryList=boundary_list,
                              outputFilename=filename_syll_lab,
                                label=label)

            print(i_boundary)
            print(len(obs_i))
            # print(np.array(groundtruth_syllable)*fs/hopsize)

            if varin['plot']:
                nestedUL = nestedUtteranceLists[i_line][1]
                # print(nestedUL)
                groundtruth_onset = [l[0]-line[0] for l in nestedUL]
                groundtruth_syllables = [l[2] for l in nestedUL]

                # plot Error analysis figures
                plt.figure(figsize=(16, 6))
                # plt.figure(figsize=(8, 4))
                # class weight
                ax1 = plt.subplot(3,1,1)
                y = np.arange(0, 80)
                x = np.arange(0, mfcc_line.shape[0])*hopsize_t
                cax = plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 11:80 * 12]))
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


if __name__ == '__main__':

    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtistAlbumFilter()

    # nacta2017
    onsetFunctionAllRecordings(wav_path=nacta2017_wav_path,
                               textgrid_path=nacta2017_textgrid_path,
                               score_path=nacta2017_score_pinyin_path,
                               test_recordings=testNacta2017,
                               feature_type='mfccBands2D',
                               dmfcc=False,
                               nbf=True,
                               mth=mth_ODF,
                               late_fusion=fusion)

    # nacta
    onsetFunctionAllRecordings(wav_path=nacta_wav_path,
                               textgrid_path=nacta_textgrid_path,
                               score_path=nacta_score_pinyin_path,
                               test_recordings=testNacta,
                               feature_type='mfccBands2D',
                               dmfcc=False,
                               nbf=True,
                               mth=mth_ODF,
                               late_fusion=fusion)

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