#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle,cPickle,gzip

import numpy as np
from sklearn import mixture,preprocessing
from sklearn.model_selection import train_test_split
import essentia.standard as ess
import h5py

from parameters import *
from phonemeMap import *
from textgridParser import textGrid2WordList, wordListsParseByLines, syllableTextgridExtraction
from scoreParser import csvDurationScoreParser
from trainTestSeparation import getTestTrainRecordingsMaleFemale, \
                            getTestTrainrecordingsRiyaz, \
                            getTestTrainRecordingsNactaISMIR, \
                            getTestTrainRecordingsArtist, \
                            getTestTrainRecordingsArtistAlbumFilter
from Fdeltas import Fdeltas
from Fprev_sub import Fprev_sub
from file_path_jingju import *
from labParser import lab2WordList
from utilFunctions import featureReshape

from audio_preprocessing import _nbf_2D
from audio_preprocessing import getMFCCBands2DMadmom


def getFeature(audio, d=True, nbf=False):

    '''
    MFCC of give audio interval [p[0],p[1]]
    :param audio:
    :param p:
    :return:
    '''

    winAnalysis = 'hann'

    # this MFCC is for pattern classification, which numberBands always be by default
    MFCC40 = ess.MFCC(sampleRate=fs,
                      highFrequencyBound=highFrequencyBound,
                      inputSize=framesize + 1)

    N = 2 * framesize  # padding 1 time framesize
    SPECTRUM = ess.Spectrum(size=N)
    WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC40(mXFrame)
        # mfccFrame       = mfccFrame[1:]
        mfcc.append(mfccFrame)

    if d:
        mfcc            = np.array(mfcc).transpose()
        dmfcc           = Fdeltas(mfcc,w=5)
        ddmfcc          = Fdeltas(dmfcc,w=5)
        feature         = np.transpose(np.vstack((mfcc,dmfcc,ddmfcc)))
    else:
        feature         = np.array(mfcc)

    if not d and nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_out = np.array(mfcc, copy=True)
        for w_r in range(1,6):
            mfcc_right_shifted = Fprev_sub(mfcc, w=w_r)
            mfcc_left_shifted = Fprev_sub(mfcc, w=-w_r)
            mfcc_out = np.vstack((mfcc_out, mfcc_left_shifted, mfcc_right_shifted))
        feature = np.array(np.transpose(mfcc_out),dtype='float32')

    # print feature.shape

    return feature

def getMFCCBands1D(audio, nbf=False):

    '''
    mel bands feature [p[0],p[1]], this function only for pdnn acoustic model training
    output feature is a 1d vector
    it needs the array format float32
    :param audio:
    :param p:
    :param nbf: bool, if we need to neighbor frames
    :return:
    '''

    winAnalysis = 'hann'

    MFCC80 = ess.MFCC(sampleRate=fs,
                      highFrequencyBound=highFrequencyBound,
                      inputSize=framesize + 1,
                      numberBands=80)

    N = 2 * framesize  # padding 1 time framesize
    SPECTRUM = ess.Spectrum(size=N)
    WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC80(mXFrame)
        mfcc.append(bands)

    if nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_right_shifted_1 = Fprev_sub(mfcc, w=1)
        mfcc_left_shifted_1 = Fprev_sub(mfcc, w=-1)
        mfcc_right_shifted_2 = Fprev_sub(mfcc, w=2)
        mfcc_left_shifted_2 = Fprev_sub(mfcc, w=-2)
        feature = np.transpose(np.vstack((mfcc,
                                          mfcc_right_shifted_1,
                                          mfcc_left_shifted_1,
                                          mfcc_right_shifted_2,
                                          mfcc_left_shifted_2)))
    else:
        feature = mfcc

    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature


def getMFCCBands2D(audio, framesize_t, hopsize_t, fs, nbf=False, nlen=10):

    '''
    mel bands feature [p[0],p[1]]
    output feature for each time stamp is a 2D matrix
    it needs the array format float32
    :param audio:
    :param p:
    :param nbf: bool, if we need neighbor frames
    :return:
    '''

    winAnalysis = 'hann'

    highFrequencyBound = fs / 2 if fs / 2 < 11000 else 11000

    framesize = int(round(framesize_t * fs))
    hopsize = int(round(hopsize_t * fs))

    MFCC80 = ess.MFCC(sampleRate=fs,
                      highFrequencyBound=highFrequencyBound,
                      inputSize=framesize + 1,
                      numberBands=80)

    N = 2 * framesize  # padding 1 time framesize
    SPECTRUM = ess.Spectrum(size=N)
    WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC80(mXFrame)
        mfcc.append(bands)

    if nbf:
        feature = _nbf_2D(mfcc, nlen)
    else:
        feature = mfcc
    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature

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

def getMBE(audio):
    '''
    mel band energy feature
    :param audio:
    :return:
    '''

    winAnalysis = 'hann'

    # this MFCC is for pattern classification, which numberBands always be by default
    MFCC40 = ess.MFCC(sampleRate=fs,
                      highFrequencyBound=highFrequencyBound,
                      inputSize=framesize + 1)

    N = 2 * framesize  # padding 1 time framesize
    SPECTRUM = ess.Spectrum(size=N)
    WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

    mfccBands = []
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):

        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC40(mXFrame)
        mfccBands.append(bands)
    feature         = np.array(mfccBands)
    return feature

def featureLabelOnset(mfcc_p, mfcc_n, scaling=False):
    '''
    organize the training feature and label
    :param
    :return:
    '''

    label_p = [1] * mfcc_p.shape[0]
    label_n = [0] * mfcc_n.shape[0]

    feature_all = np.concatenate((mfcc_p, mfcc_n), axis=0)
    label_all = label_p+label_n

    feature_all = np.array(feature_all,dtype='float32')
    label_all = np.array(label_all,dtype='int64')

    # scaler = preprocessing.StandardScaler()
    # scaler.fit(feature_all)
    #
    # if scaling:
    #     feature_all = scaler.transform(feature_all)

    scaler = None

    return feature_all, label_all, scaler

def featureLabelOnsetH5py(filename_mfcc_p, filename_mfcc_n, scaling=True):
    '''
    organize the training feature and label
    :param
    :return:
    '''

    # feature_all = np.concatenate((mfcc_p, mfcc_n), axis=0)
    f_mfcc_p = h5py.File(filename_mfcc_p, 'a')
    f_mfcc_n = h5py.File(filename_mfcc_n, 'r')

    dim_p_0 = f_mfcc_p['mfcc_p'].shape[0]
    dim_n_0 = f_mfcc_n['mfcc_n'].shape[0]
    dim_1 = f_mfcc_p['mfcc_p'].shape[1]

    label_p = [1] * dim_p_0
    label_n = [0] * dim_n_0
    label_all = label_p + label_n

    feature_all = np.zeros((dim_p_0+dim_n_0, dim_1), dtype='float32')

    print('concatenate features... ...')

    feature_all[:dim_p_0, :] = f_mfcc_p['mfcc_p']
    feature_all[dim_p_0:, :] = f_mfcc_n['mfcc_n']

    f_mfcc_p.flush()
    f_mfcc_p.close()
    f_mfcc_n.flush()
    f_mfcc_n.close()

    label_all = np.array(label_all,dtype='int64')

    print('scaling features... ... ')

    scaler = preprocessing.StandardScaler()
    scaler.fit(feature_all)
    if scaling:
        feature_all = scaler.transform(feature_all)

    return feature_all, label_all, scaler

def featureLabelPhoneme(dic_pho_feature_train):
    '''
    organize the training feature and label
    :param dic_pho_feature_train: input dictionary, key: phoneme, value: feature vectors
    :return:
    '''
    feature_all = []
    label_all = []
    for key in dic_pho_feature_train:
        feature = dic_pho_feature_train[key]
        label = [dic_pho_label[key]] * len(feature)

        if len(feature):
            if not len(feature_all):
                feature_all = feature
            else:
                feature_all = np.vstack((feature_all, feature))
            label_all += label
    label_all = np.array(label_all,dtype='int64')

    scaler = preprocessing.StandardScaler()
    scaler.fit(feature_all)
    feature_all = scaler.transform(feature_all)

    return feature_all, label_all, scaler

def dumpFeaturePhoneme(full_path_recordings,
                       full_path_textgrids,
                       syllableTierName,
                       phonemeTierName,
                       feature_type='mfcc',
                       dmfcc=True,
                       nbf=False):
    '''
    dump the MFCC for each phoneme
    :param recordings:
    :return:
    '''

    ##-- dictionary feature
    dic_pho_feature = {}

    for _,pho in enumerate(set(dic_pho_map.values())):
        dic_pho_feature[pho] = np.array([])

    for ii_rec, recording in enumerate(full_path_recordings):

        lineList = textGrid2WordList(full_path_textgrids[ii_rec], whichTier=syllableTierName)
        utteranceList = textGrid2WordList(full_path_textgrids[ii_rec], whichTier=phonemeTierName)

        # parse lines of groundtruth
        nestedPhonemeLists, _, _ = wordListsParseByLines(lineList, utteranceList)

        # audio
        wav_full_filename   = recording
        audio               = ess.MonoLoader(downmix = 'left', filename = wav_full_filename, sampleRate = fs)()

        if feature_type == 'mfcc':
            # MFCC feature
            mfcc = getFeature(audio, d=dmfcc, nbf=nbf)
        elif feature_type == 'mfccBands1D':
            mfcc = getMFCCBands1D(audio, nbf=nbf)
            mfcc = np.log(100000*mfcc+1)
        elif feature_type == 'mfccBands2D':
            mfcc = getMFCCBands2D(audio, framesize_t, hopsize_t, fs, nbf=nbf, nlen=varin['nlen'])
            mfcc = np.log(100000*mfcc+1)
        else:
            print(feature_type+' is not exist.')
            raise

        for ii,pho in enumerate(nestedPhonemeLists):
            print 'calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists))
            for p in pho[1]:
                # map from annotated xsampa to readable notation
                key = dic_pho_map[p[2]]

                sf = int(round(p[0]*fs/float(hopsize))) # starting frame
                ef = int(round(p[1]*fs/float(hopsize))) # ending frame

                mfcc_p = mfcc[sf:ef,:]  # phoneme syllable

                if not len(dic_pho_feature[key]):
                    dic_pho_feature[key] = mfcc_p
                else:
                    dic_pho_feature[key] = np.vstack((dic_pho_feature[key],mfcc_p))

    return dic_pho_feature

def complicateSampleWeighting(mfcc, frames_onset, frame_start, frame_end):
    """
    Weight +/-6 frames around the onset frame, first 3 frames as the positive weighting.
    :param mfcc:
    :param frames_onset:
    :param frame_start:
    :param frame_end:
    :return:
    """
    frames_onset_p75 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p50 = np.hstack((frames_onset - 2, frames_onset + 2))
    frames_onset_p25 = np.hstack((frames_onset - 3, frames_onset + 3))

    frames_onset_p75 = removeOutOfRange(frames_onset_p75, frame_start, frame_end)
    frames_onset_p50 = removeOutOfRange(frames_onset_p50, frame_start, frame_end)
    frames_onset_p25 = removeOutOfRange(frames_onset_p25, frame_start, frame_end)
    # print(frames_onset_p75, frames_onset_p50, frames_onset_p25)

    # mfcc positive
    mfcc_p100 = mfcc[frames_onset, :]
    mfcc_p75 = mfcc[frames_onset_p75, :]
    mfcc_p50 = mfcc[frames_onset_p50, :]
    mfcc_p25 = mfcc[frames_onset_p25, :]

    # print(mfcc_p100.shape, mfcc_p75.shape, mfcc_p50.shape)

    frames_n25 = np.hstack((frames_onset - 4, frames_onset + 4))
    frames_n50 = np.hstack((frames_onset - 5, frames_onset + 5))
    frames_n75 = np.hstack((frames_onset - 6, frames_onset + 6))

    frames_n25 = removeOutOfRange(frames_n25, frame_start, frame_end)
    frames_n50 = removeOutOfRange(frames_n50, frame_start, frame_end)
    frames_n75 = removeOutOfRange(frames_n75, frame_start, frame_end)

    # mfcc negative
    mfcc_n25 = mfcc[frames_n25, :]
    mfcc_n50 = mfcc[frames_n50, :]
    mfcc_n75 = mfcc[frames_n75, :]

    frames_all = np.arange(frame_start, frame_end)
    frames_n100 = np.setdiff1d(frames_all, np.hstack((frames_onset,
                                                      frames_onset_p75,
                                                      frames_onset_p50,
                                                      frames_onset_p25,
                                                      frames_n25,
                                                      frames_n50,
                                                      frames_n75)))
    # print(frames_n100.shape, frames_all.shape)
    mfcc_n100 = mfcc[frames_n100, :]

    mfcc_p = np.concatenate((mfcc_p100, mfcc_p75, mfcc_p50, mfcc_p25), axis=0)
    sample_weights_p = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                       np.ones((mfcc_p75.shape[0],)) * 0.75,
                                       np.ones((mfcc_p50.shape[0],)) * 0.5,
                                       np.ones((mfcc_p25.shape[0],)) * 0.25))
    # print(sample_weights_p)
    # print(mfcc_p.shape)

    mfcc_n = np.concatenate((mfcc_n100, mfcc_n75, mfcc_n50, mfcc_n25), axis=0)
    sample_weights_n = np.concatenate((np.ones((mfcc_n100.shape[0],)),
                                       np.ones((mfcc_n75.shape[0],)) * 0.75,
                                       np.ones((mfcc_n50.shape[0],)) * 0.5,
                                       np.ones((mfcc_n25.shape[0],)) * 0.25))


    # mfcc_p_all.append(mfcc_p)
    # mfcc_n_all.append(mfcc_n)
    # sample_weights_p_all.append(sample_weights_p)
    # sample_weights_n_all.append(sample_weights_n)
    return mfcc_p, mfcc_n, sample_weights_p, sample_weights_n


def positiveThreeSampleWeighting(mfcc, frames_onset, frame_start, frame_end):
    """
    Weight +/-6 frames around the onset frame, first 3 frames as the positive weighting.
    :param mfcc:
    :param frames_onset:
    :param frame_start:
    :param frame_end:
    :return:
    """
    frames_onset_p75 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p50 = np.hstack((frames_onset - 2, frames_onset + 2))
    frames_onset_p25 = np.hstack((frames_onset - 3, frames_onset + 3))

    frames_onset_p75 = removeOutOfRange(frames_onset_p75, frame_start, frame_end)
    frames_onset_p50 = removeOutOfRange(frames_onset_p50, frame_start, frame_end)
    frames_onset_p25 = removeOutOfRange(frames_onset_p25, frame_start, frame_end)
    # print(frames_onset_p75, frames_onset_p50, frames_onset_p25)

    # mfcc positive
    mfcc_p100 = mfcc[frames_onset, :]
    mfcc_p75 = mfcc[frames_onset_p75, :]
    mfcc_p50 = mfcc[frames_onset_p50, :]
    mfcc_p25 = mfcc[frames_onset_p25, :]

    # print(mfcc_p100.shape, mfcc_p75.shape, mfcc_p50.shape)


    frames_all = np.arange(frame_start, frame_end)
    frames_n100 = np.setdiff1d(frames_all, np.hstack((frames_onset,
                                                      frames_onset_p75,
                                                      frames_onset_p50,
                                                      frames_onset_p25)))
    # print(frames_n100.shape, frames_all.shape)
    mfcc_n100 = mfcc[frames_n100, :]

    mfcc_p = np.concatenate((mfcc_p100, mfcc_p75, mfcc_p50, mfcc_p25), axis=0)
    sample_weights_p = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                       np.ones((mfcc_p75.shape[0],)) * 0.75,
                                       np.ones((mfcc_p50.shape[0],)) * 0.5,
                                       np.ones((mfcc_p25.shape[0],)) * 0.25))
    # print(sample_weights_p)
    # print(mfcc_p.shape)

    mfcc_n = mfcc_n100
    sample_weights_n = np.ones((mfcc_n100.shape[0],))


    # mfcc_p_all.append(mfcc_p)
    # mfcc_n_all.append(mfcc_n)
    # sample_weights_p_all.append(sample_weights_p)
    # sample_weights_n_all.append(sample_weights_n)
    return mfcc_p, mfcc_n, sample_weights_p, sample_weights_n

def simpleSampleWeighting(mfcc, frames_onset, frame_start, frame_end):
    """
    simple weighing strategy used in Schluter's paper
    :param mfcc:
    :param frames_onset:
    :param frame_start:
    :param frame_end:
    :return:
    """

    frames_onset_p25 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p25 = removeOutOfRange(frames_onset_p25, frame_start, frame_end)
    # print(frames_onset_p75, frames_onset_p50, frames_onset_p25)

    # mfcc positive
    mfcc_p100 = mfcc[frames_onset, :]
    mfcc_p25 = mfcc[frames_onset_p25, :]

    frames_all = np.arange(frame_start, frame_end)
    frames_n100 = np.setdiff1d(frames_all, np.hstack((frames_onset,
                                                      frames_onset_p25)))
    # print(frames_n100.shape, frames_all.shape)
    mfcc_n100 = mfcc[frames_n100, :]

    mfcc_p = np.concatenate((mfcc_p100, mfcc_p25), axis=0)
    sample_weights_p = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                       np.ones((mfcc_p25.shape[0],)) * 0.25))
    # print(sample_weights_p)
    # print(mfcc_p.shape)

    mfcc_n = mfcc_n100
    sample_weights_n = np.ones((mfcc_n100.shape[0],))

    return mfcc_p, mfcc_n, sample_weights_p, sample_weights_n



def removeOutOfRange(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]


def dumpFeatureOnsetHelper(lab,
                           wav_path,
                           textgrid_path,
                           score_path,
                           artist_name,
                           recording_name,
                           feature_type,
                           nbf):
    if not lab:
        groundtruth_textgrid_file = os.path.join(textgrid_path, artist_name, recording_name + '.TextGrid')
        print(groundtruth_textgrid_file)
        wav_file = os.path.join(wav_path, artist_name, recording_name + '.wav')
    else:
        groundtruth_textgrid_file = os.path.join(textgrid_path, artist_name, recording_name + '.lab')
        wav_file = os.path.join(wav_path, artist_name, recording_name + '.mp3')

    if '2017' in artist_name:
        score_file = os.path.join(score_path, artist_name, recording_name + '.csv')
    else:
        score_file = os.path.join(score_path, artist_name, recording_name + '.csv')

    # if not os.path.isfile(score_file):
    #     print 'Score not found: ' + score_file
    #     continue

    if not lab:
        lineList = textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
        utteranceList = textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

        # parse lines of groundtruth
        nestedUtteranceLists, numLines, numUtterances = wordListsParseByLines(lineList, utteranceList)
    else:
        nestedUtteranceLists = [lab2WordList(groundtruth_textgrid_file, label=True)]

    # parse score
    _, utterance_durations, bpm = csvDurationScoreParser(score_file)

    # load audio
    fs = 44100

    if feature_type != 'madmom':
        if not lab:
            audio = ess.MonoLoader(downmix='left', filename=wav_file, sampleRate=fs)()
        else:
            audio, fs, nc, md5, br, codec = ess.AudioLoader(filename=wav_file)()
            audio = audio[:, 0]  # take the left channel

    if feature_type == 'mfccBands2D':
        mfcc = getMFCCBands2D(audio, framesize_t, hopsize_t, fs, nbf=nbf, nlen=varin['nlen'])
        mfcc = np.log(100000 * mfcc + 1)
    elif feature_type == 'madmom':
        mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
    else:
        print(feature_type + ' is not exist.')
        raise
    return nestedUtteranceLists, utterance_durations, bpm, mfcc


def getFrameOnset(recording_name, idx, lab, u_list):
    print 'Processing feature collecting ... ' + recording_name + ' phrase ' + str(idx + 1)

    if not lab:
        times_onset = [u[0] for u in u_list[1]]
    else:
        times_onset = [u[0] for u in u_list]

    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / hopsize_t), dtype=int)

    # line start and end frames
    frame_start = frames_onset[0]
    if not lab:
        frame_end = int(u_list[0][1] / hopsize_t)
    else:
        frame_end = int(u_list[-1][1] / hopsize_t)
    return frames_onset, frame_start, frame_end


def dumpFeatureOnset(wav_path,
                     textgrid_path,
                     score_path,
                     recordings,
                     feature_type='mfcc',
                     dmfcc=True,
                     nbf=False,
                     lab=False,
                     sampleWeighting='complicate'):
    '''
    dump the MFCC for each phoneme
    :param recordings:
    :return:
    '''

    # p: position, n: negative, 75: 0.75 sample_weight
    mfcc_p_all = []
    mfcc_n_all = []
    sample_weights_p_all = []
    sample_weights_n_all = []

    for artist_name, recording_name in recordings:

        nestedUtteranceLists, utterance_durations, bpm, mfcc = dumpFeatureOnsetHelper(lab,
                                                                                      wav_path,
                                                                                      textgrid_path,
                                                                                      score_path,
                                                                                      artist_name,
                                                                                      recording_name,
                                                                                      feature_type,
                                                                                      nbf)
        # print(mfcc.shape)

        # create the ground truth lab files
        for idx, u_list in enumerate(nestedUtteranceLists):
            try:
                print(bpm[idx])
            except IndexError:
                continue


            if float(bpm[idx]):

                frames_onset, frame_start, frame_end = getFrameOnset(recording_name, idx, lab, u_list)

                if sampleWeighting == 'complicate':
                    mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = complicateSampleWeighting(mfcc, frames_onset, frame_start, frame_end)
                elif sampleWeighting == 'positiveThree':
                    print('postive three weighting')
                    mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = positiveThreeSampleWeighting(mfcc, frames_onset, frame_start, frame_end)
                else:
                    mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = simpleSampleWeighting(mfcc, frames_onset, frame_start, frame_end)

                mfcc_p_all.append(mfcc_p)
                mfcc_n_all.append(mfcc_n)
                sample_weights_p_all.append(sample_weights_p)
                sample_weights_n_all.append(sample_weights_n)

                # print(len(mfcc_p_all), len(mfcc_n_all), len(sample_weights_p_all), len(sample_weights_n_all))

    return np.concatenate(mfcc_p_all), \
           np.concatenate(mfcc_n_all), \
           np.concatenate(sample_weights_p_all), \
           np.concatenate(sample_weights_n_all)


def featureOnsetPhraseLabelSampleWeights(frames_onset, frame_start, frame_end, mfcc):
    frames_onset_p25 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p25 = removeOutOfRange(frames_onset_p25, frame_start, frame_end)

    len_line = frame_end - frame_start + 1

    mfcc_line = mfcc[frame_start:frame_end+1, :]

    sample_weights = np.ones((len_line,))
    sample_weights[frames_onset_p25 - frame_start] = 0.25

    label = np.zeros((len_line,))
    label[frames_onset - frame_start] = 1
    label[frames_onset_p25 - frame_start] = 1
    return mfcc_line, label, sample_weights

def dumpFeatureOnsetPhrase(wav_path,
                         textgrid_path,
                         score_path,
                         recordings,
                         feature_type='mfcc',
                         dmfcc=True,
                         nbf=False,
                         lab=False,
                         sampleWeighting='complicate',
                           split='ismir'):
    '''
    dump feature for each phrase
    for one dataset
    :param recordings:
    :return:
    '''
    mfcc_line_all = []
    for artist_name, recording_name in recordings:

        nestedUtteranceLists, utterance_durations, bpm, mfcc = dumpFeatureOnsetHelper(lab,
                                                                                      wav_path,
                                                                                      textgrid_path,
                                                                                      score_path,
                                                                                      artist_name,
                                                                                      recording_name,
                                                                                      feature_type,
                                                                                      nbf)

        # print(mfcc.shape)

        # create the ground truth lab files
        for idx, u_list in enumerate(nestedUtteranceLists):
            try:
                print(bpm[idx])
            except IndexError:
                continue

            if float(bpm[idx]):
                print 'Processing feature collecting ... ' + recording_name + ' phrase ' + str(idx + 1)

                frames_onset, frame_start, frame_end = getFrameOnset(recording_name, idx, lab, u_list)

                mfcc_line, label, sample_weights = \
                    featureOnsetPhraseLabelSampleWeights(frames_onset, frame_start, frame_end, mfcc)

                # save feature, label, sample weights
                feature_data_split_path = join(feature_data_path, split)
                if not os.path.exists(feature_data_split_path):
                    os.makedirs(feature_data_split_path)

                filename_mfcc_line = join(feature_data_split_path,
                                           'feature'+'_'+artist_name + '_' +recording_name+'_'+str(idx)+'.h5')
                h5f = h5py.File(filename_mfcc_line, 'w')
                h5f.create_dataset('feature_all', data=mfcc_line)
                h5f.close()

                filename_label = join(feature_data_split_path,
                                       'label'+'_'+artist_name + '_' +recording_name+'_'+str(idx)+'.pickle.gz')
                cPickle.dump(label, gzip.open(filename_label, 'wb'), cPickle.HIGHEST_PROTOCOL)

                filename_sample_weights = join(feature_data_split_path,
                                       'sample_weights' + '_' + artist_name + '_' + recording_name + '_' + str(idx) + '.pickle.gz')
                cPickle.dump(sample_weights, gzip.open(filename_sample_weights, 'wb'), cPickle.HIGHEST_PROTOCOL)

                mfcc_line_all.append(mfcc_line)

    return np.concatenate(mfcc_line_all)


def datasetSwitcher(split='ismir'):
    if split =='ismir':
        testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsNactaISMIR()
    elif split == 'artist':
        testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtist()
    elif split == 'artist_filter':
        testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtistAlbumFilter()
    else:
        testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsMaleFemale()
    return testNacta2017, testNacta, trainNacta2017, trainNacta


def dumpFeatureBatchOnset(split='ismir', feature_type='mfccBands2D', train_test = 'train', sampleWeighting='simple'):
    """
    dump features for all the dataset for onset detection
    :return:
    """

    testNacta2017, testNacta, trainNacta2017, trainNacta = datasetSwitcher(split)

    if train_test == 'train':
        nacta_data = trainNacta
        nacta_data_2017 = trainNacta2017
        scaling = True
    else:
        nacta_data = testNacta
        nacta_data_2017 = testNacta2017
        scaling = False

    if len(trainNacta2017):
        mfcc_p_nacta2017, \
        mfcc_n_nacta2017, \
        sample_weights_p_nacta2017, \
        sample_weights_n_nacta2017 \
            = dumpFeatureOnset(wav_path=nacta2017_wav_path,
                               textgrid_path=nacta2017_textgrid_path,
                               score_path=nacta2017_score_path,
                               recordings=nacta_data_2017,
                               feature_type=feature_type,
                               dmfcc=False,
                               nbf=True,
                               sampleWeighting=sampleWeighting)

    mfcc_p_nacta, \
    mfcc_n_nacta, \
    sample_weights_p_nacta, \
    sample_weights_n_nacta \
        = dumpFeatureOnset(wav_path=nacta_wav_path,
                           textgrid_path=nacta_textgrid_path,
                           score_path=nacta_score_path,
                           recordings=nacta_data,
                           feature_type=feature_type,
                           dmfcc=False,
                           nbf=True,
                           sampleWeighting=sampleWeighting)

    print(trainNacta2017, testNacta2017)

    print('finished feature extraction.')

    if len(trainNacta2017):
        mfcc_p = np.concatenate((mfcc_p_nacta2017, mfcc_p_nacta))
        mfcc_n = np.concatenate((mfcc_n_nacta2017, mfcc_n_nacta))
        sample_weights_p = np.concatenate((sample_weights_p_nacta2017, sample_weights_p_nacta))
        sample_weights_n = np.concatenate((sample_weights_n_nacta2017, sample_weights_n_nacta))
    else:
        mfcc_p = mfcc_p_nacta
        mfcc_n = mfcc_n_nacta
        sample_weights_p = sample_weights_p_nacta
        sample_weights_n = sample_weights_n_nacta

    sample_weights = np.concatenate((sample_weights_p, sample_weights_n))

    filename_mfcc_p = join(feature_data_path, 'mfcc_p_' + split + '_split.h5')
    h5f = h5py.File(filename_mfcc_p, 'w')
    h5f.create_dataset('mfcc_p', data=mfcc_p)
    h5f.close()

    filename_mfcc_n = join(feature_data_path, 'mfcc_n_' + split + '_split.h5')
    h5f = h5py.File(filename_mfcc_n, 'w')
    h5f.create_dataset('mfcc_n', data=mfcc_n)
    h5f.close()

    del mfcc_p
    del mfcc_n

    feature_all, label_all, scaler = featureLabelOnsetH5py(filename_mfcc_p, filename_mfcc_n, scaling=scaling)

    os.remove(filename_mfcc_p)
    os.remove(filename_mfcc_n)

    if train_test == 'train':
        if feature_type != 'madmom':
            nlen = 10
        else:
            nlen = 7

        feature_all = featureReshape(feature_all, nlen=nlen)

    print('feature shape:', feature_all.shape)

    filename_feature_all = join(feature_data_path, 'feature_all_' + split + '_' + feature_type+'_'+train_test+'_'+sampleWeighting + '.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    print('finished feature concatenation.')

    cPickle.dump(label_all,
                 gzip.open(
                     '../trainingData/labels_'+train_test+'_set_all_syllableSeg_mfccBands2D_old+new_' + split + '_' + feature_type+'_'+sampleWeighting + '.pickle.gz',
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    if train_test == 'train':
        cPickle.dump(sample_weights,
                     gzip.open('../trainingData/sample_weights_syllableSeg_mfccBands2D_old+new_' + split + '_' + feature_type+'_'+sampleWeighting + '.pickle.gz',
                               'wb'), cPickle.HIGHEST_PROTOCOL)

        print(sample_weights_p.shape, sample_weights_n.shape)

        pickle.dump(scaler,
                    open('../cnnModels/scaler_syllable_mfccBands2D_old+new_'+split+'_' + feature_type+'_'+sampleWeighting +'.pkl', 'wb'))


def dumpFeatureBatchOnsetPhrase(split='ismir', feature_type='mfccBands2D', train_test='train'):
    """
    dump features for each phrase (line)
    :return:
    """

    testNacta2017, testNacta, trainNacta2017, trainNacta = datasetSwitcher(split)

    if train_test == 'train':
        nacta_data = trainNacta
        nacta_data_2017 = trainNacta2017
    else:
        nacta_data = testNacta
        nacta_data_2017 = testNacta2017

    if len(trainNacta2017):

        mfcc_line_nacta2017 = dumpFeatureOnsetPhrase(wav_path=nacta2017_wav_path,
                                                       textgrid_path=nacta2017_textgrid_path,
                                                       score_path=nacta2017_score_path,
                                                       recordings=nacta_data_2017,
                                                       feature_type=feature_type,
                                                       dmfcc=False,
                                                       nbf=True,
                                                       sampleWeighting='simple',
                                                       split=split)

    mfcc_line_nacta = dumpFeatureOnsetPhrase(wav_path=nacta_wav_path,
                                               textgrid_path=nacta_textgrid_path,
                                               score_path=nacta_score_path,
                                               recordings=nacta_data,
                                               feature_type=feature_type,
                                               dmfcc=False,
                                               nbf=True,
                                               sampleWeighting='simple',
                                               split=split)

    print(trainNacta2017, testNacta2017)

    print('finished feature extraction.')

    if len(trainNacta2017):
        mfcc_all = np.concatenate((mfcc_line_nacta2017, mfcc_line_nacta))
    else:
        mfcc_all = mfcc_line_nacta

    if train_test == 'train':
        scaler = preprocessing.StandardScaler()
        scaler.fit(mfcc_all)

        filename_scaler = '../cnnModels/scaler_syllable_mfccBands2D_old+new_' + split + '_' + feature_type + '_' + 'phrase' + '.pkl'
        pickle.dump(scaler, open(filename_scaler, 'wb'))


def dumpFeatureBatchOnsetTest():
    """
    dump features for the test dataset for onset detection
    :return:
    """
    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtistAlbumFilter()

    mfcc_p_nacta, \
    mfcc_n_nacta, \
    sample_weights_p_nacta, \
    sample_weights_n_nacta \
        = dumpFeatureOnset(wav_path=nacta_wav_path,
                           textgrid_path=nacta_textgrid_path,
                           score_path=nacta_score_path,
                           recordings=testNacta,
                           feature_type='mfccBands2D',
                           dmfcc=False,
                           nbf=True)

    mfcc_p_nacta1017, \
    mfcc_n_nacta2017, \
    sample_weights_p_nacta2017, \
    sample_weights_n_nacta2017 \
        = dumpFeatureOnset(wav_path=nacta2017_wav_path,
                           textgrid_path=nacta2017_textgrid_path,
                           score_path=nacta2017_score_path,
                           recordings=testNacta2017,
                           feature_type='mfccBands2D',
                           dmfcc=False,
                           nbf=True)

    print('finished feature extraction.')

    mfcc_p = np.concatenate((mfcc_p_nacta1017, mfcc_p_nacta))
    mfcc_n = np.concatenate((mfcc_n_nacta2017, mfcc_n_nacta))

    print('finished feature concatenation.')

    feature_all, label_all, scaler = featureLabelOnset(mfcc_p, mfcc_n, scaling=False)

    print(mfcc_p.shape, mfcc_n.shape)

    h5f = h5py.File('../trainingData/feature_test_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_split.h5', 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    cPickle.dump(label_all,
                 gzip.open(
                     '../trainingData/label_test_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_split.pickle.gz',
                     'wb'), cPickle.HIGHEST_PROTOCOL)


def dumpFeatureBatchOnsetRiyaz():
    """
    dump feature for Riyaz dataset
    :return:
    """
    testRiyaz, trainRiyaz = getTestTrainrecordingsRiyaz()

    mfcc_p, \
    mfcc_n, \
    sample_weights_p, \
    sample_weights_n \
        = dumpFeatureOnset(wav_path=riyaz_mp3_path,
                           textgrid_path=riyaz_groundtruthlab_path,
                           score_path=riyaz_score_path,
                           recordings=trainRiyaz,
                           feature_type='mfccBands2D',
                           dmfcc=False,
                           nbf=True,
                           lab=True)

    sample_weights = np.concatenate((sample_weights_p, sample_weights_n))

    feature_all, label_all, scaler = featureLabelOnset(mfcc_p, mfcc_n)

    print(mfcc_p.shape, mfcc_n.shape, sample_weights_p.shape, sample_weights_n.shape)

    pickle.dump(scaler, open('../cnnModels/scaler_syllable_mfccBands2D_riyaz'+str(varin['nlen'])+'.pkl', 'wb'))

    feature_all = featureReshape(feature_all, nlen=varin['nlen'])

    print(feature_all.shape)

    h5f = h5py.File(join(riyaz_feature_data_path, 'feature_all_riyaz'+str(varin['nlen'])+'.h5'), 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    cPickle.dump(label_all,
                 gzip.open('../trainingData/labels_train_set_all_syllableSeg_mfccBands2D_riyaz'+str(varin['nlen'])+'.pickle.gz', 'wb'),
                 cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights,
                 gzip.open('../trainingData/sample_weights_syllableSeg_mfccBands2D_riyaz'+str(varin['nlen'])+'.pickle.gz', 'wb'),
                 cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    # processAcousticModelTrain(mode=dataset,
    #                           syllableTierName=syllableTierName,
    #                           phonemeTierName=phonemeTierName,
    #                           featureFilename='dic_pho_feature_train_'+dataset+'.pkl',
    #                           gmmModel_path=gmmModel_path)

    # dump feature for DNN training, with getFeature output MFCC bands
    dumpFeatureBatchOnset(split='artist_filter', feature_type='madmom', train_test='train', sampleWeighting='positiveThree')
    # dumpFeatureBatchOnset(split='artist_filter', feature_type='madmom', train_test='train', sampleWeighting='positiveThree')
    # dumpFeatureBatchOnsetPhrase(split='artist_filter', feature_type='madmom', train_test='train')
    # dumpFeatureBatchOnsetRiyaz()
    # dumpFeatureBatchOnsetTest()
    # testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordings()
    #
    # for artist_path, filename in testNacta:
    #     print(join(artist_path,filename))
