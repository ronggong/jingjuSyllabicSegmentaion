#!/usr/bin/env python
# -*- coding: utf-8 -*-

import essentia.standard as ess
from schluterParser import annotationCvParser
from trainTestSeparation import getRecordings
from trainingSampleCollection import getMFCCBands2D, removeOutOfRange, featureLabelOnset
from parameters import *
from filePathSchulter import *
import numpy as np
import h5py
import cPickle, gzip
from multiprocessing import Process


def dumpFeatureOnset(audio_path, annotation_path, fn, dmfcc=True, nbf=False, lab=False):
    '''
    dump the MFCC for each phoneme
    :param recordings:
    :return:
    '''

    # p: position, n: negative, 75: 0.75 sample_weight


    # annotation_filenames = getRecordings(annotation_path)
    # for fn in annotation_filenames:


    audio_fn = join(audio_path, fn+'.flac')
    annotation_fn = join(annotation_path, fn+'.onsets')

    # if not os.path.isfile(score_file):
    #     print 'Score not found: ' + score_file
    #     continue

    # load audio
    # fs = 44100

    # if not lab:
    #     audio               = ess.MonoLoader(downmix = 'left', filename = wav_file, sampleRate = fs)()
    # else:
    audio, fs, nc, md5, br, codec = ess.AudioLoader(filename = audio_fn)()
    audio = audio[:,0] # take the left channel

    mfcc = getMFCCBands2D(audio, framesize_t, hopsize_t, fs, nbf=nbf, nlen=varin['nlen'])
    mfcc = np.log(100000*mfcc+1)
    # print('mfcc', mfcc.shape)

    print 'Feature collecting ... ' + fn

    times_onset = annotationCvParser(annotation_fn)
    times_onset = [float(to) for to in times_onset]
    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset)/hopsize_t),dtype=int)
    # print(frames_onset)
    # print(mfcc.shape)

    # line start and end frames
    # frame_start = frames_onset[0]
    frame_start = 0
    frame_end   = int(len(audio)/float(fs)/hopsize_t)

    print(frame_start, frame_end)

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


    # mfcc_p_all.append(mfcc_p)
    # mfcc_n_all.append(mfcc_n)
    # sample_weights_p_all.append(sample_weights_p)
    # sample_weights_n_all.append(sample_weights_n)

    # print(len(mfcc_p_all), len(mfcc_n_all), len(sample_weights_p_all), len(sample_weights_n_all))
    feature_all, label_all, scaler = featureLabelOnset(mfcc_p, mfcc_n, scaling=False)

    sample_weights = np.concatenate((sample_weights_p, sample_weights_n))

    # print(feature_all.shape, label_all.shape, sample_weights.shape)

    filename_feature_all = join(schluter_feature_data_path, 'feature_'+fn+'.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    cPickle.dump(label_all,
                 gzip.open(
                     join(schluter_feature_data_path, 'label_'+fn+'.pickle.gz'),
                     'wb'), cPickle.HIGHEST_PROTOCOL)


    cPickle.dump(sample_weights,
                 gzip.open(
                     join(schluter_feature_data_path, 'sample_weights_'+fn+'.pickle.gz'),
                     'wb'), cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    for fn in getRecordings(schluter_annotations_path):
        # if ii < 21840:
        #     continue
        p = Process(target=dumpFeatureOnset,
                    args=(schluter_audio_path, schluter_annotations_path,fn,True,True,))
        p.start()
        p.join()
