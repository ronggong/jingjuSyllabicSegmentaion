'''
 * Copyright (C) 2017  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuSingingPhraseMatching
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''

# this file is reserved for other usage

import pickle

import essentia.standard as ess
import numpy as np

from src.file_path_jingju import *
from src.parameters import *
from src.utilFunctions import featureReshape


def pitchProcessing_audio(audio):
    N           = 2 * framesize  # padding 1 time framesize
    SPECTRUM    = ess.Spectrum(size=N)
    WINDOW      = ess.Windowing(type='blackmanharris62', zeroPadding=N - framesize)
    PITCHYIN    = ess.PitchYinFFT(frameSize=N, sampleRate = fs)

    pitch = []
    pitchConfidence = []
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame       = WINDOW(frame)
        mXFrame     = SPECTRUM(frame)
        pitchFrame, pitchConfidenceFrame  = PITCHYIN(mXFrame)
        pitch.append(pitchFrame)
        pitchConfidence.append(pitchConfidenceFrame)

    # discard pitch below 65, higher than 1000 Hz, confidence below 0.85
    index_keep = discardFrameByConfidence(pitch, pitchConfidence, 65, 1000, 0.85)

    return index_keep

def discardFrameByConfidence(pitch, pitchConfidence, low_threshold_pitch, high_threshold_pitch, threshold_confidence):
    '''
    keep the pitch if confidence > threshold and pitch > low_threshold and pitch < high_threshold
    '''
    index_keep = []
    for ii in range(len(pitch)):
        if not (pitchConfidence[ii] > threshold_confidence
                and pitch[ii] > low_threshold_pitch
                and pitch[ii] < high_threshold_pitch):
            index_keep.append(ii)

    return index_keep

def discardFrameByOnsetFunction(onset, th=0.1):
    index_keep = []
    for ii in range(len(onset)):
        if onset[ii] > th:
            index_keep.append(ii)

    return index_keep


def indexKeepDictionary(index_keep):
    """
    dictionary map result index to original index
    :param index_keep:
    :return:
    """
    dict_index_keep = {}
    for ii in range(len(index_keep)):
        dict_index_keep[ii] = index_keep[ii]
    return dict_index_keep

def mfccFeatureScaler(feature, feature_type='mfccBands2D'):
    """
    feature scaling only for acoustic model observation
    :param feature:
    :param feature_type:
    :return:
    """

    if feature_type == 'mfccBands2D':
        scaler  = pickle.load(open(full_path_mfccBands_2D_scaler_am,'rb'))
        feature_scaled = scaler.transform(feature)

    return feature_scaled


def featureIndexKeep(feature, index_keep):

    feature_out         = feature[index_keep[0],:]

    for index in index_keep[1:]:
        feature_out = np.vstack((feature_out,feature[index,:]))

    return feature_out


def processFeature(audio,onset_function,feature,feature_type='mfccBands2D'):
    index_keep_confidence   = pitchProcessing_audio(audio)
    index_keep_onset        = discardFrameByOnsetFunction(onset_function, th=0.1)
    index_keep              = sorted(list(set(index_keep_confidence+index_keep_onset)))
    index_keep              = [ik for ik in index_keep if ik < len(onset_function)]
    dict_index_keep         = indexKeepDictionary(index_keep)
    feature                 = mfccFeatureScaler(feature, feature_type)
    feature_out             = featureIndexKeep(feature, index_keep)

    if feature_type == 'mfccBands2D':
        feature_out = featureReshape(feature_out)

    return feature_out, dict_index_keep