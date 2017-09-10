#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle,cPickle,gzip

import numpy as np
from sklearn import mixture,preprocessing
from sklearn.model_selection import train_test_split
import essentia.standard as ess

from src.parameters import *
from src.phonemeMap import *
from src.textgridParser import textGrid2WordList, wordListsParseByLines, syllableTextgridExtraction
from src.scoreParser import csvDurationScoreParser
from src.trainTestSeparation import getTestTrainRecordings
from src.Fdeltas import Fdeltas
from src.Fprev_sub import Fprev_sub
from src.filePath import *


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

def getMFCCBands2D(audio, framesize, nbf=False, nlen=10):

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
        mfcc_out = np.array(mfcc, copy=True)
        for ii in range(1,nlen+1):
            mfcc_right_shift    = Fprev_sub(mfcc, w=ii)
            mfcc_left_shift     = Fprev_sub(mfcc, w=-ii)
            mfcc_out = np.vstack((mfcc_right_shift, mfcc_out, mfcc_left_shift))
        feature = mfcc_out.transpose()
    else:
        feature = mfcc
    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature

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

def featureLabelOnset(mfcc_p, mfcc_n):
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

    scaler = preprocessing.StandardScaler()
    scaler.fit(feature_all)
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

def featureReshape(feature, nlen=10):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen*2+1

    feature_reshaped = np.zeros((n_sample,n_row,n_col),dtype='float32')
    print("reshaping feature...")
    for ii in range(n_sample):
        # print ii
        feature_frame = np.zeros((n_row,n_col),dtype='float32')
        for jj in range(n_col):
            feature_frame[:,jj] = feature[ii][n_row*jj:n_row*(jj+1)]
        feature_reshaped[ii,:,:] = feature_frame
    return feature_reshaped

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
            mfcc = getMFCCBands2D(audio, framesize, nbf=nbf, nlen=varin['nlen'])
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


def dumpFeatureBatchPhoneme():
    """
    batch processing for collecting phoneme features
    :return:
    """
    # recordings = getRecordingNames('TRAIN')
    full_path_recordings    = []
    full_path_textgrids     = []

    for rec in queenMary_Recordings_train_fem:
        full_path_recordings.append(os.path.join(aCapella_root, queenMarydataset_path, audio_path, rec+'.wav'))
        full_path_textgrids.append(os.path.join(textgrid_path_dan, dict_name_mapping_dan_qm[rec]+'.textgrid'))

    for rec in queenMary_Recordings_train_male:
        full_path_recordings.append(os.path.join(aCapella_root, queenMarydataset_path, audio_path, rec+'.wav'))
        full_path_textgrids.append(os.path.join(textgrid_path_laosheng, dict_name_mapping_laosheng_qm[rec]+'.textgrid'))

    for rec in bcn_Recordings_train_male:
        full_path_recordings.append(os.path.join(aCapella_root, bcnRecording_path, audio_path, rec+'.wav'))
        full_path_textgrids.append(os.path.join(textgrid_path_laosheng, dict_name_mapping_laosheng_bcn[rec]+'.textgrid'))

    for rec in bcn_Recordings_train_fem:
        full_path_recordings.append(os.path.join(aCapella_root, bcnRecording_path, audio_path, rec+'.wav'))
        full_path_textgrids.append(os.path.join(textgrid_path_dan, dict_name_mapping_dan_bcn[rec]+'.textgrid'))

    for rec in london_Recordings_train_male:
        full_path_recordings.append(os.path.join(aCapella_root, londonRecording_path, audio_path, rec + '.wav'))
        full_path_textgrids.append(os.path.join(textgrid_path_laosheng, dict_name_mapping_laosheng_london[rec] + '.textgrid'))

    for rec in london_Recordings_train_fem:
        full_path_recordings.append(os.path.join(aCapella_root, londonRecording_path, audio_path, rec + '.wav'))
        full_path_textgrids.append(os.path.join(textgrid_path_dan, dict_name_mapping_dan_london[rec] + '.textgrid'))

    dic_pho_feature = dumpFeaturePhoneme(full_path_recordings,
                                       full_path_textgrids,
                                       syllableTierName='dian',
                                       phonemeTierName='details',
                                       feature_type='mfccBands2D',
                                       dmfcc=False,
                                       nbf=True)

    feature_all, label_all, scaler = featureLabelPhoneme(dic_pho_feature)


    pickle.dump(scaler, open('scaler_syllableSeg_am_mfccBands2D.pkl', 'wb'))

    feature_all = featureReshape(feature_all, nlen=varin['nlen'])

    cPickle.dump((feature_all, label_all),
                 gzip.open('train_set_all_syllableSeg_am_mfccBands2D.pickle.gz', 'wb'),
                 cPickle.HIGHEST_PROTOCOL)

    feature_train, feature_validation, label_train, label_validation = \
        train_test_split(feature_all, label_all, test_size=0.2, stratify=label_all)

    # -- dump feature vectors training and validation sets separately
    cPickle.dump((feature_train, label_train),
                 gzip.open('train_set_syllableSeg_am_mfccBands2D.pickle.gz', 'wb'),
                 cPickle.HIGHEST_PROTOCOL)
    cPickle.dump((feature_validation, label_validation),
                 gzip.open('validation_set_syllableSeg_am_mfccBands2D.pickle.gz', 'wb'),
                 cPickle.HIGHEST_PROTOCOL)

    print feature_train.shape, len(feature_validation), len(label_train), len(label_validation)


def removeOutOfRange(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]


def dumpFeatureOnset(wav_path, textgrid_path, score_path, recordings, feature_type='mfcc', dmfcc=True, nbf=False):
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
        groundtruth_textgrid_file   = os.path.join(textgrid_path, artist_name, recording_name+'.TextGrid')
        wav_file                    = os.path.join(wav_path, artist_name, recording_name+'.wav')

        if '2017' in artist_name:
            score_file = os.path.join(score_path, artist_name, recording_name+'.csv')
        else:
            score_file = os.path.join(score_path, recording_name+'.csv')

        # if not os.path.isfile(score_file):
        #     print 'Score not found: ' + score_file
        #     continue

        lineList = textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
        utteranceList = textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

        # parse lines of groundtruth
        nestedUtteranceLists, numLines, numUtterances = wordListsParseByLines(lineList, utteranceList)

        # parse score
        utterance_durations, bpm = csvDurationScoreParser(score_file)

        # load audio
        audio               = ess.MonoLoader(downmix = 'left', filename = wav_file, sampleRate = fs)()

        if feature_type == 'mfcc':
            # MFCC feature
            mfcc = getFeature(audio, d=dmfcc, nbf=nbf)
        elif feature_type == 'mfccBands1D':
            mfcc = getMFCCBands1D(audio, nbf=nbf)
            mfcc = np.log(100000*mfcc+1)
        elif feature_type == 'mfccBands2D':
            mfcc = getMFCCBands2D(audio, framesize, nbf=nbf, nlen=varin['nlen'])
            mfcc = np.log(100000*mfcc+1)
        else:
            print(feature_type+' is not exist.')
            raise

        # create the ground truth lab files
        for idx, list in enumerate(nestedUtteranceLists):
            try:
                print(bpm[idx])
            except IndexError:
                continue


            if float(bpm[idx]):
                print 'Processing feature collecting ... ' + recording_name + ' phrase ' + str(idx + 1)

                times_onset = [u[0] for u in list[1]]
                # syllable onset frames
                frames_onset = np.array(np.around(np.array(times_onset)*fs/hopsize),dtype=int)

                # line start and end frames
                frame_start = frames_onset[0]
                frame_end   = int(list[0][1]*fs/hopsize)

                frames_onset_p75 = np.hstack((frames_onset-1, frames_onset+1))
                frames_onset_p50 = np.hstack((frames_onset - 2, frames_onset + 2))
                frames_onset_p25 = np.hstack((frames_onset - 3, frames_onset + 3))

                frames_onset_p75 = removeOutOfRange(frames_onset_p75, frame_start, frame_end)
                frames_onset_p50 = removeOutOfRange(frames_onset_p50, frame_start, frame_end)
                frames_onset_p25 = removeOutOfRange(frames_onset_p25, frame_start, frame_end)
                # print(frames_onset_p75, frames_onset_p50, frames_onset_p25)

                # mfcc positive
                mfcc_p100 = mfcc[frames_onset,:]
                mfcc_p75 = mfcc[frames_onset_p75,:]
                mfcc_p50 = mfcc[frames_onset_p50,:]
                mfcc_p25 = mfcc[frames_onset_p25,:]

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

                frames_all = np.arange(frame_start,frame_end)
                frames_n100   = np.setdiff1d(frames_all,np.hstack((frames_onset,
                                               frames_onset_p75,
                                               frames_onset_p50,
                                               frames_onset_p25,
                                               frames_n25,
                                               frames_n50,
                                               frames_n75)))
                # print(frames_n100.shape, frames_all.shape)
                mfcc_n100     = mfcc[frames_n100,:]

                mfcc_p = np.concatenate((mfcc_p100,mfcc_p75,mfcc_p50,mfcc_p25),axis=0)
                sample_weights_p = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                                 np.ones((mfcc_p75.shape[0],))*0.75,
                                                 np.ones((mfcc_p50.shape[0],))*0.5,
                                                 np.ones((mfcc_p25.shape[0],))*0.25))
                # print(sample_weights_p)
                # print(mfcc_p.shape)

                mfcc_n = np.concatenate((mfcc_n100,mfcc_n75,mfcc_n50,mfcc_n25),axis=0)
                sample_weights_n = np.concatenate((np.ones((mfcc_n100.shape[0],)),
                                                 np.ones((mfcc_n75.shape[0],))*0.75,
                                                 np.ones((mfcc_n50.shape[0],))*0.5,
                                                 np.ones((mfcc_n25.shape[0],))*0.25))

                mfcc_p_all.append(mfcc_p)
                mfcc_n_all.append(mfcc_n)
                sample_weights_p_all.append(sample_weights_p)
                sample_weights_n_all.append(sample_weights_n)

                # print(len(mfcc_p_all), len(mfcc_n_all), len(sample_weights_p_all), len(sample_weights_n_all))

    return np.concatenate(mfcc_p_all), \
           np.concatenate(mfcc_n_all), \
           np.concatenate(sample_weights_p_all), \
           np.concatenate(sample_weights_n_all)

def dumpFeatureBatchOnset():
    """
    dump features for all the dataset for onset detection
    :return:
    """
    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordings()

    mfcc_p_nacta1017, \
    mfcc_n_nacta2017, \
    sample_weights_p_nacta2017, \
    sample_weights_n_nacta2017 \
        = dumpFeatureOnset(wav_path=nacta2017_wav_path,
                           textgrid_path=nacta2017_textgrid_path,
                           score_path=nacta2017_score_path,
                           recordings=trainNacta2017,
                           feature_type='mfccBands2D',
                           dmfcc=False,
                           nbf=True)

    mfcc_p_nacta, \
    mfcc_n_nacta, \
    sample_weights_p_nacta, \
    sample_weights_n_nacta \
        = dumpFeatureOnset(wav_path=nacta_wav_path,
                           textgrid_path=nacta_textgrid_path,
                           score_path=nacta_score_path,
                           recordings=trainNacta,
                           feature_type='mfccBands2D',
                           dmfcc=False,
                           nbf=True)

    print('finished feature extraction.')

    mfcc_p = np.concatenate((mfcc_p_nacta1017, mfcc_p_nacta))
    mfcc_n = np.concatenate((mfcc_n_nacta2017, mfcc_n_nacta))
    sample_weights_p = np.concatenate((sample_weights_p_nacta2017, sample_weights_p_nacta))
    sample_weights_n = np.concatenate((sample_weights_n_nacta2017, sample_weights_n_nacta))

    print('finished feature concatenation.')

    sample_weights = np.concatenate((sample_weights_p, sample_weights_n))

    feature_all, label_all, scaler = featureLabelOnset(mfcc_p, mfcc_n)

    print(mfcc_p.shape, mfcc_n.shape, sample_weights_p.shape, sample_weights_n.shape)

    pickle.dump(scaler, open('cnnModels/scaler_syllable_mfccBands2D_old+new.pkl', 'wb'))

    feature_all = featureReshape(feature_all, nlen=varin['nlen'])

    print(feature_all.shape)

    for ii in range(feature_all.shape[0]):
        print('dumping feature', ii)
        cPickle.dump(feature_all[ii,:,:],
                     gzip.open('trainingData/features_train_set_all_syllableSeg_mfccBands2D_old+new/'+str(ii)+'.pickle.gz', 'wb'),
                     cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(label_all,
                 gzip.open('trainingData/labels_train_set_all_syllableSeg_mfccBands2D_old+new.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights,
                 gzip.open('trainingData/sample_weights_syllableSeg_mfccBands2D_old+new.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    # processAcousticModelTrain(mode=dataset,
    #                           syllableTierName=syllableTierName,
    #                           phonemeTierName=phonemeTierName,
    #                           featureFilename='dic_pho_feature_train_'+dataset+'.pkl',
    #                           gmmModel_path=gmmModel_path)

    # dump feature for DNN training, with getFeature output MFCC bands
    dumpFeatureBatchOnset()