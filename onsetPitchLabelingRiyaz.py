# -*- coding: utf-8 -*-
import pickle
from os import makedirs
from os.path import isfile, exists

import essentia.standard as ess
import numpy as np
import pyximport
from keras.models import load_model

pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from src.filePath import *
from src.labWriter import boundaryLabWriter
from src.labParser import lab2WordList
from src.parameters import *
from src.scoreParser import csvDurationScoreParser
from src.pitchCalculation import pitchCalculation
from src.utilFunctions import featureReshape
from datasetCollection.trainingSampleCollection import getMFCCBands2D
from datasetCollection.trainingSampleCollection import getTestTrainrecordingsRiyaz

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
        mfcc_reshaped = featureReshape(mfcc_scaled, nlen=varin['nlen'])
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

def pitchLabeling(audio,
                  frameSize,
                  sampleRate,
                  maxFrequency,
                  phrase_start_time,
                  time_boundary_start,
                  time_boundary_end):

    list_pitch = []

    if time_boundary_start[0] > 2*(frameSize/float(sampleRate)):
        p, pc = pitchCalculation(audio,
                                [int(phrase_start_time*sampleRate),
                                 int((time_boundary_start[0]+phrase_start_time)*sampleRate)],
                                 frameSize,
                                 sampleRate,
                                 maxFrequency)
        list_pitch.append([0.0, p, pc])
        print('before segment:', p, pc)
    for ii in range(len(time_boundary_start)):
        p, pc = pitchCalculation(audio,
                             [int((time_boundary_start[ii]+phrase_start_time) * sampleRate),
                              int((time_boundary_end[ii]+phrase_start_time) * sampleRate)],
                             frameSize,
                             sampleRate,
                             maxFrequency)
        print('segment:', ii, p, pc)
        list_pitch.append([time_boundary_start[ii], p, pc])
    return list_pitch


def onsetFunctionAllRecordings(wav_path,
                               textgrid_path,
                               score_path,
                               test_recordings,
                               feature_type='mfcc',
                               dmfcc=False,
                               nbf=True,
                               mth='jordi',
                               late_fusion=True,
                               lab=False,
                               decoding_method='viterbi'):
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


    for artist_path, rn in test_recordings:
        # rn = rn.split('.')[0]

        groundtruth_textgrid_file   = join(textgrid_path, artist_path, rn+'.lab')
        wav_file = join(wav_path, artist_path, rn + '.mp3')

        score_file                  = join(score_path, artist_path, rn+'.csv')

        if not isfile(score_file):
            print 'Score not found: ' + score_file
            continue

        lineList        = [lab2WordList(groundtruth_textgrid_file, label=True)]
        syllables, syllable_durations, bpm = csvDurationScoreParser(score_file)

        # print(pinyins)
        # print(syllable_durations)


        audio, fs, nc, md5, br, codec = ess.AudioLoader(filename=wav_file)()
        audio_monoloader = audio[:, 0]  # take the left channel

        mfcc, mfcc_reshaped = featureExtraction(audio_monoloader,
                                                      scaler,
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

            frame_start = int(round(line[0][0] / hopsize_t))
            frame_end = int(round(line[-1][1] /hopsize_t))
            # print(feature.shape)

            mfcc_line          = mfcc[frame_start:frame_end]
            mfcc_reshaped_line = mfcc_reshaped[frame_start:frame_end]

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

            obs_i = np.squeeze(obs_i)

            print(obs_i.shape)
            # organize score
            print('Calculating: '+rn+' phrase '+str(i_obs))
            print('ODF Methods: '+mth_ODF+' Late fusion: '+str(fusion))

            time_line      = line[-1][1] - line[0][0]

            duration_score = syllable_durations[i_line]
            duration_score = np.array([float(ds) for ds in duration_score if len(ds)])
            duration_score = duration_score * (time_line/np.sum(duration_score))

            # print(duration_score)

            if decoding_method == 'viterbi':
                # segmental decoding
                obs_i[0] = 1.0
                obs_i[-1] = 1.0
                i_boundary = viterbiDecoding.viterbiSegmental2(obs_i, duration_score, varin)
                filename_syll_lab = join(eval_results_path, artist_path, rn + '_' + str(i_line + 1) + '.syll.lab')
            else:
                i_boundary = peakPicking(obs_i)
                filename_syll_lab = join(eval_results_path+'_peakPicking', artist_path, rn + '_' + str(i_line + 1) + '.syll.lab')

            time_boundray_start = np.array(i_boundary[:-1])*hopsize_t
            time_boundray_end   = np.array(i_boundary[1:])*hopsize_t

            # list_pitch = pitchLabeling(audio_monoloader,
            #                           int(framesize_t*fs),
            #                           fs,
            #                           2000,
            #                           line[0][0],
            #                           time_boundray_start,
            #                           time_boundray_end)

            # uncomment this section if we want to write boundaries to .syll.lab file

            eval_results_data_path = dirname(filename_syll_lab)

            if not exists(eval_results_data_path):
                makedirs(eval_results_data_path)

            # write boundary lab file

            # boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist(), syllables[i_line])
            boundary_list = zip(time_boundray_start.tolist(), time_boundray_end.tolist())
            label = False

            boundaryLabWriter(boundaryList=boundary_list,
                              outputFilename=filename_syll_lab,
                                label=label)

            print(i_boundary)
            print(len(obs_i))
            # print(np.array(groundtruth_syllable)*fs/hopsize)

            if varin['plot']:
                groundtruth_onset = [l[0]-line[0][0] for l in line]
                groundtruth_syllables = [l[2] for l in line]

                # plot Error analysis figures
                plt.figure(figsize=(16, 6))
                # plt.figure(figsize=(8, 4))
                # class weight
                ax1 = plt.subplot(3,1,1)
                y = np.arange(0, 80)
                x = np.arange(0, mfcc_line.shape[0])*hopsize_t
                cax = plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * varin['nlen']:80 * (varin['nlen']+1)]))
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
                for p in list_pitch:
                    if p[2] > 0.3:
                        plt.text(p[0], ax2.get_ylim()[1]*0.8, str(int(p[1])))


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

    # Riyaz
    testRiyaz, trainRiyaz = getTestTrainrecordingsRiyaz()

    onsetFunctionAllRecordings(wav_path=riyaz_mp3_path,
                               textgrid_path=riyaz_groundtruthlab_path,
                               score_path=riyaz_score_path,
                               test_recordings=testRiyaz,
                               feature_type='mfccBands2D',
                               dmfcc=False,
                               nbf=True,
                               mth=mth_ODF,
                               late_fusion=fusion,
                               lab=True,
                               decoding_method='viterbi')
