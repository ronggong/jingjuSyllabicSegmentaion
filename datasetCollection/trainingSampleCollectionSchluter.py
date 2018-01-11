#!/usr/bin/env python
# -*- coding: utf-8 -*-

import essentia.standard as ess
from schluterParser import annotationCvParser
from utilFunctions import getRecordings
from trainingSampleCollection import getMFCCBands2D, featureLabelOnset
from trainingSampleCollection import getMFCCBands2DMadmom
from trainingSampleCollection import simpleSampleWeighting, complicateSampleWeighting, positiveThreeSampleWeighting
from trainingSampleCollection import featureOnsetPhraseLabelSampleWeights
from parameters import *
from filePathSchulter import *
import numpy as np
import h5py
import cPickle, gzip
from multiprocessing import Process
import os


def dumpFeatureOnsetHelper(audio_path, annotation_path, fn, method, nbf, channel):

    audio_fn = join(audio_path, fn + '.flac')
    annotation_fn = join(annotation_path, fn + '.onsets')

    audio, fs, nc, md5, br, codec = ess.AudioLoader(filename=audio_fn)()

    if method == 'essentia':
        audio = audio[:, 0]  # take the left channel
        mfcc = getMFCCBands2D(audio, framesize_t, hopsize_t, fs, nbf=nbf, nlen=varin['nlen'])
        mfcc = np.log(100000 * mfcc + 1)
        # print('mfcc', mfcc.shape)
    else:
        mfcc = getMFCCBands2DMadmom(audio_fn, fs, hopsize_t, channel)
    # print(mfcc.shape)

    print 'Feature collecting ... ' + fn

    times_onset = annotationCvParser(annotation_fn)
    times_onset = [float(to) for to in times_onset]
    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / hopsize_t), dtype=int)
    # print(frames_onset)
    # print(mfcc.shape)

    # line start and end frames
    # frame_start = frames_onset[0]
    frame_start = 0
    frame_end = mfcc.shape[0] - 1
    return mfcc, frames_onset, frame_start, frame_end


def featureDataPathSwitcher(method, sampleWeighting, channel):

    if method == 'essentia':
        feature_data_path = schluter_feature_data_path
    else:
        if sampleWeighting == 'complicate':
            feature_data_path = schluter_feature_data_path_madmom
        elif sampleWeighting == 'positiveThree':
            feature_data_path = schluter_feature_data_path_madmom_positiveThreeSampleWeighting
        else:
            if channel == 1:
                feature_data_path = schluter_feature_data_path_madmom_simpleSampleWeighting
            else:
                feature_data_path = schluter_feature_data_path_madmom_simpleSampleWeighting_3channel

    return feature_data_path


def featureLabelWeightsSaver(feature_data_path, feature_all, label_all, sample_weights):

    filename_feature_all = join(feature_data_path, 'feature_' + fn + '.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    cPickle.dump(label_all,
                 gzip.open(
                     join(feature_data_path, 'label_' + fn + '.pickle.gz'),
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights,
                 gzip.open(
                     join(feature_data_path, 'sample_weights_' + fn + '.pickle.gz'),
                     'wb'), cPickle.HIGHEST_PROTOCOL)

def dumpFeatureOnset(audio_path,
                     annotation_path,
                     fn,
                     nbf=False,
                     lab=False,
                     method='essentia',
                     channel = 1,
                     sampleWeighting='complicate'):
    '''
    dump feature, label and sample weights
    :param recordings:
    :return:
    '''


    # annotation_filenames = getRecordings(annotation_path)
    # for fn in annotation_filenames:


    mfcc, frames_onset, frame_start, frame_end = dumpFeatureOnsetHelper(audio_path, annotation_path, fn, method, nbf, channel)

    print(frame_start, frame_end)

    if sampleWeighting == 'complicate':
        mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = complicateSampleWeighting(mfcc, frames_onset, frame_start, frame_end)
    elif sampleWeighting == 'positiveThree':
        mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = positiveThreeSampleWeighting(mfcc, frames_onset, frame_start, frame_end)
    else:
        print('simple weighting...')
        mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = simpleSampleWeighting(mfcc, frames_onset, frame_start, frame_end)

    # print(len(mfcc_p_all), len(mfcc_n_all), len(sample_weights_p_all), len(sample_weights_n_all))
    feature_all, label_all, scaler = featureLabelOnset(mfcc_p, mfcc_n, scaling=False)

    sample_weights = np.concatenate((sample_weights_p, sample_weights_n))

    # print(feature_all.shape, label_all.shape, sample_weights.shape)

    feature_data_path = featureDataPathSwitcher(method, sampleWeighting, channel)

    # print(feature_data_path)

    featureLabelWeightsSaver(feature_data_path, feature_all, label_all, sample_weights)


def dumpFeatureOnsetPhrase(audio_path,
                             annotation_path,
                             fn,
                             nbf=False,
                             lab=False,
                             method='essentia',
                             channel = 1,
                             sampleWeighting='complicate'):
    '''
    dump feature, label and sample weights in phrase level, in order
    :param recordings:
    :return:
    '''


    # annotation_filenames = getRecordings(annotation_path)
    # for fn in annotation_filenames:


    mfcc, frames_onset, frame_start, frame_end = dumpFeatureOnsetHelper(audio_path, annotation_path, fn, method, nbf, channel)

    mfcc_line, label, sample_weights = featureOnsetPhraseLabelSampleWeights(frames_onset, frame_start, frame_end, mfcc)


    # print(len(mfcc_p_all), len(mfcc_n_all), len(sample_weights_p_all), len(sample_weights_n_all))
    # print(feature_all.shape, label_all.shape, sample_weights.shape)

    # print(feature_data_path)

    feature_data_path = schluter_feature_data_path_madmom_simpleSampleWeighting_phrase

    if not os.path.exists(feature_data_path):
        os.makedirs(feature_data_path)

    featureLabelWeightsSaver(feature_data_path, mfcc_line, label, sample_weights)

    return mfcc_line

if __name__ == '__main__':
    for fn in getRecordings(schluter_annotations_path):

        dumpFeatureOnset(schluter_audio_path,
                       schluter_annotations_path,
                       fn,
                       True,
                       False,
                       'madmom',
                       1,
                        'positiveThree')

    # from sklearn import preprocessing
    # import pickle
    #
    # mfcc_line_all = []
    # for fn in getRecordings(schluter_annotations_path):
    #     # if ii < 21840:
    #     #     continue
    #     mfcc_line = dumpFeatureOnsetPhrase(schluter_audio_path,
    #                                        schluter_annotations_path,
    #                                        fn,
    #                                        True,
    #                                        False,
    #                                        'madmom',
    #                                        1,
    #                                        'simple')
    #     mfcc_line_all.append(mfcc_line)
    #
    # mfcc_line_all = np.concatenate(mfcc_line_all)
    #
    # scaler = preprocessing.StandardScaler()
    # scaler.fit(mfcc_line_all)
    #
    # filename_scaler = 'cnnModels/scaler_syllable_mfccBands2D_schluter_madmom' + '_' + 'phrase' + '.pkl'
    # pickle.dump(scaler, open(filename_scaler, 'wb'))
