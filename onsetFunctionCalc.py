# -*- coding: utf-8 -*-
import pickle
from os import makedirs
from os.path import isfile, exists, dirname, join

import essentia.standard as ess
import numpy as np
from keras.models import load_model

import pyximport


from src.filePath import *
from src.labWriter import boundaryLabWriter
from src.parameters import *
from src.scoreManip import phonemeDurationForLine
from src.scoreParser import generatePinyin
from src.textgridParser import textGrid2WordList, wordListsParseByLines
from trainingSampleCollection import featureReshape
from trainingSampleCollection import getMFCCBands2D
from peakPicking import peakPicking

pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

from viterbiDecoding import viterbiSegmental2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

cnnModel_name = 'jordi_temporal_old_ismir'
cnnModel_name_1 = 'jordi_timbral_old_ismir'

# print(full_path_keras_cnn_0)
# model_keras_cnn_0 = load_model(full_path_keras_cnn_0)

def getOnsetFunction(observations, model, method='jan'):
    """
    Load CNN model to calculate ODF
    :param observations:
    :return:
    """

    # ##-- call pdnn to calculate the observation from the features
    # if method=='jordi':
    #     observations = [observations, observations, observations, observations, observations, observations]
    # elif method=='jordi_horizontal_timbral':
    #     observations = [observations, observations, observations, observations, observations, observations,
    #                     observations, observations, observations, observations, observations, observations]
    #
    # obs = model.predict_proba(observations, batch_size=128)

    if method == 'jordi':
        obs = model.predict(observations, batch_size=128, verbose=1)
    else:
        obs = model.predict_proba(observations, batch_size=128)
    return obs


def featureExtraction(audio_monoloader, scaler, framesize, dmfcc=False, nbf=False, feature_type='mfccBands2D'):
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
        mfcc = getMFCCBands2D(audio_monoloader, framesize, nbf=nbf, nlen=varin['nlen'])
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


def onsetFunctionAllRecordings(recordings,
                               textgrid_path,
                               dict_recording_name_mapping,
                               dataset_path,
                               feature_type='mfcc',
                               dmfcc=False,
                               nbf=True,
                               mth='jordi',
                               late_fusion=True):
    """
    ODF and viter decoding
    :param recordings:
    :param textgrid_path:
    :param dict_recording_name_mapping: mapping from "fem_01" to standard format, see filePath.py
    :param dataset_path:
    :param feature_type: 'mfcc', 'mfccBands1D' or 'mfccBands2D'
    :param dmfcc: delta for 'mfcc'
    :param nbf: context frames
    :param mth: jordi, jordi_horizontal_timbral, jan, jan_chan3
    :param late_fusion: Bool
    :return:
    """

    scaler = pickle.load(open(full_path_mfccBands_2D_scaler_onset, 'rb'))

    # kerasModel = _LRHMM.kerasModel(full_path_keras_cnn_am)

    for i_recording, recording_name in enumerate(recordings):

        groundtruth_textgrid_file   = join(textgrid_path, dict_recording_name_mapping[recording_name]+'.TextGrid')
        score_file                  = join(aCapella_root, dataset_path, score_path,      recording_name+'.csv')
        wav_file                    = join(aCapella_root, dataset_path, audio_path,      recording_name+'.wav')

        if not isfile(score_file):
            print 'Score not found: ' + score_file
            continue

        lineList        = textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
        utteranceList   = textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

        # parse lines of groundtruth
        nestedUtteranceLists, numLines, numUtterances = wordListsParseByLines(lineList, utteranceList)

        # parse score
        syllables, pinyins, syllable_durations, bpm = generatePinyin(score_file)

        # print(pinyins)
        # print(syllable_durations)

        if varin['obs'] == 'tocal':
            # load audio
            audio_monoloader               = ess.MonoLoader(downmix = 'left', filename = wav_file, sampleRate = fs)()
            audio_eqloudloder              = ess.EqloudLoader(filename=wav_file, sampleRate = fs)()

            if mth == 'jordi' or mth == 'jordi_horizontal_timbral' or mth == 'jan':
                mfcc, mfcc_reshaped = featureExtraction(audio_monoloader,
                                                              scaler,
                                                              int(round(0.025 * fs)),
                                                              dmfcc=dmfcc,
                                                              nbf=nbf,
                                                              feature_type='mfccBands2D')

        for i_obs, lineList in enumerate(nestedUtteranceLists):
            if int(bpm[i_obs]):
                sample_start    = int(round(lineList[0][0] * fs))
                sample_end      = int(round(lineList[0][1] * fs))
                frame_start     = int(round(lineList[0][0] * fs / hopsize))
                frame_end       = int(round(lineList[0][1] * fs / hopsize))
                # print(feature.shape)

                obs_path = join('./obs', cnnModel_name, dataset_path)
                obs_filename = recording_name + '_' + str(i_obs + 1) + '.pkl'
                full_obs_name = join(obs_path, obs_filename)

                if varin['obs'] == 'tocal':
                    if mth == 'jordi' or mth == 'jordi_horizontal_timbral' or mth == 'jan':
                        audio_eqloudloder_line = audio_eqloudloder[sample_start:sample_end]
                        mfcc_line          = mfcc[frame_start:frame_end]
                        mfcc_reshaped_line = mfcc_reshaped[frame_start:frame_end]

                    mfcc_reshaped_line = np.expand_dims(mfcc_reshaped_line, axis=1)
                    obs     = getOnsetFunction(observations=mfcc_reshaped_line,
                                               model=model_keras_cnn_0,
                                               method=mth)
                    # obs_i   = obs[:,1]
                    obs_i = obs[:, 0]

                    hann = np.hanning(5)
                    hann /= np.sum(hann)

                    obs_i = np.convolve(hann, obs_i, mode='same')

                    # save onset curve
                    print('save onset curve ... ...')
                    obs_dirpath = dirname(full_obs_name)
                    if not exists(obs_dirpath):
                        makedirs(obs_dirpath)
                    pickle.dump(obs_i, open(full_obs_name, 'w'))
                else:
                    obs_i = pickle.load(open(full_obs_name, 'r'))

                if late_fusion:
                    if varin['obs'] == 'viterbi':
                        obs_2 = getOnsetFunction(observations=mfcc_reshaped_line,
                                                 path_keras_cnn=full_path_keras_cnn_1,
                                                 method=mth)
                        obs_2_i = obs_2[:, 1]
                        obs_2_i = np.convolve(hann, obs_2_i, mode='same')
                    else:
                        obs_path_1 = join('./obs', cnnModel_name_1, dataset_path)
                        full_obs_name_1 = join(obs_path_1, obs_filename)
                        obs_2_i = pickle.load(open(full_obs_name_1, 'r'))

                    obs_i = late_fusion_calc(obs_i, obs_2_i, mth=2)

                # organize score
                print('Calculating: '+recording_name+' phrase '+str(i_obs))
                print('ODF Methods: '+mth_ODF+' Late fusion: '+str(fusion))

                time_line      = lineList[0][1] - lineList[0][0]

                lyrics_line    = [ll[2] for ll in lineList[1]]
                groundtruth_syllable = [ll[0]-lineList[0][0] for ll in lineList[1]]

                print('Syllable:')
                print(lyrics_line)

                print('Length of syllables, length of ground truth syllables:')
                print(len(lyrics_line), len(groundtruth_syllable))

                pinyin_score   = pinyins[i_obs]
                pinyin_score   = [ps for ps in pinyin_score if len(ps)]
                duration_score = syllable_durations[i_obs]
                duration_score = np.array([float(ds) for ds in duration_score if len(ds)])
                duration_score = duration_score * (time_line/np.sum(duration_score))

                if varin['decoding'] == 'viterbi':
                    # segmental decoding
                    obs_i[0] = 1.0
                    obs_i[-1] = 1.0
                    i_boundary = viterbiSegmental2(obs_i, duration_score, varin)
                    # # uncomment this section if we want to write boundaries to .syll.lab file
                    filename_syll_lab = join(eval_results_path, dataset_path, recording_name+'_'+str(i_obs+1)+'.syll.lab')
                    label = True

                else:
                    i_boundary = peakPicking(obs_i)
                    filename_syll_lab = join(eval_results_path + '_peakPicking', dataset_path,
                                             recording_name + '_' + str(i_obs + 1) + '.syll.lab')
                    label = False

                time_boundray_start = np.array(i_boundary[:-1]) * hopsize_t
                time_boundray_end = np.array(i_boundary[1:]) * hopsize_t

                eval_results_data_path = dirname(filename_syll_lab)

                if not exists(eval_results_data_path):
                    makedirs(eval_results_data_path)

                if varin['decoding'] == 'viterbi':
                    boundaryList = zip(time_boundray_start.tolist(), time_boundray_end.tolist(), lyrics_line)
                else:
                    boundaryList = zip(time_boundray_start.tolist(), time_boundray_end.tolist())

                # write boundary lab file
                boundaryLabWriter(boundaryList=boundaryList,
                                  outputFilename=filename_syll_lab,
                                    label=label)

                # print(i_boundary)
                # print(len(obs_i))
                # print(np.array(groundtruth_syllable)*fs/hopsize)

                if varin['plot']:
                    # plot Error analysis figures
                    plt.figure(figsize=(16, 6))
                    # plt.figure(figsize=(8, 4))
                    # class weight
                    ax1 = plt.subplot(3,1,1)
                    y = np.arange(0, 80)
                    x = np.arange(0, mfcc_line.shape[0])*(hopsize/float(fs))
                    cax = plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 11:80 * 12]))
                    for gs in groundtruth_syllable:
                        plt.axvline(gs, color='r', linewidth=2)
                    # cbar = fig.colorbar(cax)
                    ax1.set_ylabel('Mel bands', fontsize=12)
                    ax1.get_xaxis().set_visible(False)
                    ax1.axis('tight')
                    plt.title('Calculating: '+recording_name+' phrase '+str(i_obs))

                    ax2 = plt.subplot(312, sharex=ax1)
                    plt.plot(np.arange(0,len(obs_i))*(hopsize/float(fs)), obs_i)
                    for ib in i_boundary:
                        plt.axvline(ib * (hopsize / float(fs)), color='r', linewidth=2)

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
                    # plt.xlabel('Time (s)')
                    # plt.tight_layout()

                    plt.show()


if __name__ == '__main__':

    # testing arias
    recordings_qm_male  = ['male_01/neg_1', 'male_01/neg_2', 'male_01/pos_4', 'male_01/pos_5']
    recordings_qm_fem   = ['fem_01/pos_3', 'fem_10/pos_3']
    recordings_lon_fem  = ['Dan-03']
    recordings_bcn_male  = ['005', '008']

    # Queen Mary female
    onsetFunctionAllRecordings(recordings=recordings_qm_fem,
                               textgrid_path=textgrid_path_dan,
                               dict_recording_name_mapping=dict_name_mapping_dan_qm,
                               dataset_path=queenMarydataset_path,
                               feature_type='mfccBands2D',
                               dmfcc=False,
                               nbf=True,
                               mth=mth_ODF,
                               late_fusion=fusion)

    # London Female
    onsetFunctionAllRecordings(recordings=recordings_lon_fem,
                               textgrid_path=textgrid_path_dan,
                               dict_recording_name_mapping=dict_name_mapping_dan_london,
                               dataset_path=londonRecording_path,
                               feature_type='mfccBands2D',
                               dmfcc=False,
                               nbf=True,
                               mth=mth_ODF,
                               late_fusion=fusion)

    # Queen Mary Male
    onsetFunctionAllRecordings(recordings=recordings_qm_male,
                               textgrid_path=textgrid_path_laosheng,
                               dict_recording_name_mapping=dict_name_mapping_laosheng_qm,
                               dataset_path=queenMarydataset_path,
                               feature_type='mfccBands2D',
                               dmfcc=False,
                               nbf=True,
                               mth=mth_ODF,
                               late_fusion=fusion)

    # BCN Male
    onsetFunctionAllRecordings(recordings=recordings_bcn_male,
                               textgrid_path=textgrid_path_laosheng,
                               dict_recording_name_mapping=dict_name_mapping_laosheng_bcn,
                               dataset_path=bcnRecording_path,
                               feature_type='mfccBands2D',
                               dmfcc=False,
                               nbf=True,
                               mth=mth_ODF,
                               late_fusion=fusion)
