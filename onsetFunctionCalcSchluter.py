# -*- coding: utf-8 -*-
import cPickle
import gzip
import pickle
from os import makedirs
from os.path import exists

import essentia.standard as ess
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from madmom.features.onsets import OnsetPeakPickingProcessor
from datasetCollection.trainingSampleCollectionSchluter import getMFCCBands2DMadmom

from datasetCollection.trainingSampleCollection import featureExtraction
from eval_schluter import eval_schluter
from src.filePathSchulter import *

# from src.filePathJingju import jingju_cnn_model_path
# from src.filePath import full_path_mfccBands_2D_scaler_onset

from src.labWriter import onsetLabWriter
from src.parameters import *
from src.schluterParser import annotationCvParser
from src.utilFunctions import getOnsetFunction, late_fusion_calc, featureReshape, append_or_write


def onsetFunctionPlot(mfcc,
                      obs_i,
                      groundtruth_onset,
                      detected_onsets):

    # plot Error analysis figures
    plt.figure(figsize=(16, 6))
    # plt.figure(figsize=(8, 4))
    # class weight
    ax1 = plt.subplot(2, 1, 1)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc.shape[0]) * hopsize_t
    cax = plt.pcolormesh(x, y, np.transpose(mfcc[:, 80 * 10:80 * 11]))
    for i_gs, gs in enumerate(groundtruth_onset):
        plt.axvline(gs, color='r', linewidth=2)
        # plt.text(gs, ax1.get_ylim()[1], groundtruth_syllables[i_gs])

    # cbar = fig.colorbar(cax)
    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')
    # plt.title('Calculating: '+rn+' phrase '+str(i_obs))

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(np.arange(0, len(obs_i)) * hopsize_t, obs_i)
    # for i_ib in range(len(detected_onsets)):
    #     plt.axvline(detected_onsets[i_ib], color='r', linewidth=2)
        # plt.text(i_boundary[i_ib] * hopsize_t, ax2.get_ylim()[1], syllables[i_line][i_ib])

    ax2.set_ylabel('ODF', fontsize=12)
    ax2.axis('tight')
    plt.show()

def peakPickingDetectedOnsetSaver(pp_threshold, obs_i, model_name_0, model_name_1, filename):

    # madmom peak picking
    arg_pp = {'threshold': pp_threshold, 'smooth': 0.05, 'fps': 1. / hopsize_t, 'pre_max': hopsize_t,
              'post_max': hopsize_t, 'combine': 0}
    peak_picking = OnsetPeakPickingProcessor(**arg_pp)
    detected_onsets = peak_picking.process(obs_i)

    # save detected onsets
    filename_syll_lab = join(eval_results_path, model_name_0 + model_name_1, filename + '.syll.lab')

    # uncomment this section if we want to write boundaries to .syll.lab file
    eval_results_data_path = dirname(filename_syll_lab)

    if not exists(eval_results_data_path):
        makedirs(eval_results_data_path)

    onsetLabWriter(detected_onsets, filename_syll_lab)

    return detected_onsets


def onsetFunction(audio_path,
                  annotation_path,
                  filename,
                  scaler_0,
                  scaler_1,
                  model_keras_cnn_0,
                  model_keras_cnn_1,
                  model_name_0,
                  model_name_1,
                  pp_threshold=0.54,
                  melbank_mth='essentia',
                  dmfcc=False,
                  nbf=True,
                  late_fusion=True,
                  channel=1,
                  obs_cal='tocal'):
    """
    ODF and viterbi decoding
    :param recordings:
    :param textgrid_path:
    :param dataset_path:
    :param melbank_mth: 'mfcc', 'mfccBands1D' or 'mfccBands2D'
    :param dmfcc: delta for 'mfcc'
    :param nbf: context frames
    :param mth: jordi, jordi_horizontal_timbral, jan, jan_chan3
    :param late_fusion: Bool
    :return:
    """


    annotation_filename   = join(annotation_path, filename+'.onsets')
    audio_filename = join(audio_path, filename + '.flac')

    groundtruth_onset = annotationCvParser(annotation_filename)
    groundtruth_onset = [float(gto) for gto in groundtruth_onset]

    # print(pinyins)
    # print(syllable_durations)

    obs_path = join('./obs', model_name_0)
    obs_filename = filename + '.pkl'

    if obs_cal == 'tocal':
        if melbank_mth == 'essentia':
            audio, fs, nc, md5, br, codec = ess.AudioLoader(filename=audio_filename)()
            audio = audio[:, 0]  # take the left channel
            mfcc, mfcc_reshaped = featureExtraction(audio,
                                                  scaler_0,
                                                  framesize_t,
                                                    hopsize_t,
                                                    fs,
                                                  dmfcc=dmfcc,
                                                  nbf=nbf,
                                                  feature_type='mfccBands2D')
        else:
            if channel == 1:
                mfcc = getMFCCBands2DMadmom(audio_filename,fs=44100.0,hopsize_t=hopsize_t,channel=1)
                mfcc_scaled = scaler_0.transform(mfcc)
                mfcc_reshaped = featureReshape(mfcc_scaled, nlen=7)
            else:
                mfcc = getMFCCBands2DMadmom(audio_filename,fs=44100.0,hopsize_t=hopsize_t,channel=channel)
                mfcc_reshaped_conc = []
                for ii in range(channel):
                    mfcc_scaled = scaler_0[ii].transform(mfcc[:,:,ii])
                    mfcc_reshaped = featureReshape(mfcc_scaled, nlen=7)
                    mfcc_reshaped_conc.append(mfcc_reshaped)
                mfcc_reshaped = np.stack(mfcc_reshaped_conc, axis=3)

        # onset detection function smooth
        if channel == 1:
            mfcc_reshaped = np.expand_dims(mfcc_reshaped, axis=1)

        obs     = getOnsetFunction(observations=mfcc_reshaped,
                                   model=model_keras_cnn_0,
                                   method=no_dense_str)
        obs_i = obs[:,0]

        # save onset curve
        print('save onset curve ... ...')
        if not exists(obs_path):
            makedirs(obs_path)
        pickle.dump(obs_i, open(join(obs_path, obs_filename), 'w'))

    else:
        obs_i = pickle.load(open(join(obs_path, obs_filename), 'r'))

    if late_fusion:
        if varin['obs'] == 'tocal':
            # fuse second observation
            obs_2 = getOnsetFunction(observations=mfcc_reshaped,
                                     model=model_keras_cnn_1)
            obs_2_i = obs_2[:, 0]
            obs_2_i = np.convolve(hann, obs_2_i, mode='same')
        else:
            obs_path_2 = join('./obs', model_name_1)
            obs_filename = filename + '.pkl'
            obs_2_i = pickle.load(open(join(obs_path_2, obs_filename), 'r'))

        obs_i = late_fusion_calc(obs_i, obs_2_i, mth=2)

    obs_i = np.squeeze(obs_i)

    # print(obs_i.shape)
    # organize score

    # i_boundary = peakPicking(obs_i)
    detected_onsets = peakPickingDetectedOnsetSaver(pp_threshold, obs_i, model_name_0, model_name_1, filename)

    if varin['plot']:
        onsetFunctionPlot(mfcc,
                          obs_i,
                          groundtruth_onset,
                          detected_onsets)


def onsetFunctionPhrase(audio_path,
                        annotation_path,
                        filename,
                        scaler_0,
                        scaler_1,
                        model_keras_cnn_0,
                        model_keras_cnn_1,
                        model_name_0,
                        model_name_1,
                        stateful,
                        pp_threshold=0.54,
                        obs_cal='tocal'):
    """
    ODF and viterbi decoding phrase level model
    :param recordings:
    :param textgrid_path:
    :param dataset_path:
    :return:
    """


    annotation_filename   = join(annotation_path, filename+'.onsets')
    audio_filename = join(audio_path, filename + '.flac')

    groundtruth_onset = annotationCvParser(annotation_filename)
    groundtruth_onset = [float(gto) for gto in groundtruth_onset]

    # print(pinyins)
    # print(syllable_durations)

    obs_path = join('./obs', model_name_0)
    obs_filename = filename + '.pkl'

    if obs_cal == 'tocal':

        mfcc = getMFCCBands2DMadmom(audio_filename,fs=44100.0,hopsize_t=hopsize_t,channel=1)
        mfcc_scaled = scaler_0.transform(mfcc)

        # length of the paded sequence
        len_2_pad = int(len_seq * np.ceil(len(mfcc_scaled) / float(len_seq)))
        len_padded = len_2_pad - len(mfcc_scaled)

        # pad feature, label and sample weights
        mfcc_line_pad = np.zeros((len_2_pad, mfcc_scaled.shape[1]), dtype='float32')
        mfcc_line_pad[:mfcc_scaled.shape[0], :] = mfcc_scaled
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

    else:
        obs_i = pickle.load(open(join(obs_path, obs_filename), 'r'))

    obs_i = np.squeeze(obs_i)

    # i_boundary = peakPicking(obs_i)
    detected_onsets = peakPickingDetectedOnsetSaver(pp_threshold, obs_i, model_name_0, model_name_1, filename)

    if varin['plot']:
        onsetFunctionPlot(mfcc,
                          obs_i,
                          groundtruth_onset,
                          detected_onsets)

def writeResults2TxtSchluter(filename,
                             append_write,
                             best_th,
                             recall_precision_f1_overall):

    """
    :param filename:
    :param best_th: best threshold
    :param recall_precision_f1_overall:
    :return:
    """

    with open(filename, append_write) as f:
        recall = recall_precision_f1_overall[0]
        precision = recall_precision_f1_overall[1]
        f1 = recall_precision_f1_overall[2]

        f.write(str(best_th))
        f.write('\n')
        f.write(str(recall)+' '+str(precision)+' '+str(f1))
        f.write('\n')


def schluterEvalSubroutine(nfolds, filter_shape_0, weighting_str, pp_threshold, obs_cal):

    # overlap_str = '_overlap' if overlap else ''
    # phrase_str = '_phrase' if phrase_eval else ''
    # bidi_str = '_bidi' if bidi else ''
    # relu_str = '_relu' if relu else ''
    # deep_str = '_less_deep' if deep else ''
    # no_dense_str = '_no_dense' if no_dense else ''

    for ii in range(nfolds):
        if not phrase_eval: # not CRNN
            #TODO load jingju model and scaler
            model_name_str = 'schulter_' + \
                             filter_shape_0 + \
                             '_madmom_' + \
                             weighting_str + \
                             '_early_stopping_adam_cv' + \
                             relu_str + \
                             no_dense_str + \
                             deep_str + '_'

            # TODO only for jingju + schulter datasets trained model
            # scaler_name_0 = 'scaler_jan_madmom_simpleSampleWeighting_early_stopping_schluter_jingju_dataset_'+str(ii)+'.pickle.gz'
            scaler_name_0 = 'scaler_' + filter_shape_0 + '_madmom_' + weighting_str + '_early_stopping_' + str(
                ii) + '.pickle.gz'

        else: # CRNN
            model_name_str = 'schulter_' + filter_shape_0 + '_madmom_' + weighting_str + '_early_stopping_adam_cv_phrase' + overlap_str + bidi_str
            scaler_name_0 = 'scaler_syllable_mfccBands2D_schluter_madmom_phrase.pkl'

        # TODO load jingju model, to remove
        model_name_0 = model_name_str + str(ii)
        # model_name_0 = 'keras.cnn_syllableSeg_jan_artist_filter_less_deep0'

        print(model_name_0)

        model_name_1 = ''
        scaler_name_1 = ''

        test_cv_filename = join(schluter_cv_path, '8-fold_cv_random_' + str(ii) + '.fold')
        test_filenames = annotationCvParser(test_cv_filename)
        # print(test_filenames)

        # try:
            # model_keras_cnn_0 = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/cnnModels/schluter/simpleWeighting/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase_overlap0.h5'
        if obs_cal != 'tocal':
            model_keras_cnn_0 = None
            stateful = None
        else:
            if not phrase_eval:
                # TODO load jingju model
                model_keras_cnn_0 = load_model(join(schluter_cnn_model_path, model_name_0 + '.h5'))
                # model_keras_cnn_0 = load_model(join(jingju_cnn_model_path, model_name_0 + '.h5'))
            else:
                from training_scripts.models_CRNN import jan_original
                # initialize the model
                stateful = False if overlap else True
                input_shape = (1, len_seq, 1, 80, 15)
                model_keras_cnn_0 = jan_original(filter_density=1,
                                                 dropout=0.5,
                                                 input_shape=input_shape,
                                                 batchNorm=False,
                                                 dense_activation='sigmoid',
                                                 channel=1,
                                                 stateful=stateful,
                                                 training=False,
                                                 bidi=bidi)
                # load weights
                model_keras_cnn_0.load_weights(join(schluter_cnn_model_path, model_name_0 + '.h5'))
            # except:
            #     print(model_name_0, 'not found')
            #     continue

        try:
            model_keras_cnn_1 = load_model(join(schluter_cnn_model_path, model_name_1 + '.h5'))
        except:
            print(model_name_1, 'not found')
            model_keras_cnn_1 = ''

        # try:
        if not phrase_eval:
            # TODO load jingju scaler
            with gzip.open(join(schluter_cnn_model_path, scaler_name_0), 'rb') as f:
                scaler_0 = cPickle.load(f)
            # scaler_0 = pickle.load(open(full_path_mfccBands_2D_scaler_onset, 'rb'))
        else:
            scaler_0 = pickle.load(open(join(schluter_cnn_model_path, scaler_name_0), 'rb'))
        # except:
        #     print(scaler_name_0, 'not found')
        #     continue

        try:
            with gzip.open(join(schluter_cnn_model_path, scaler_name_1), 'rb') as f:
                scaler_1 = cPickle.load(f)
        except:
            print(scaler_name_1, 'not found')
            scaler_1 = ''

        # TODO load jingju model to remove
        # model_name_str = 'schluter_'+model_name_0+'_'
        # model_name_0 = 'schluter_'+model_name_0+'_'+str(ii)

        for fn in test_filenames:
            # print(fn)
            # try:
            if not phrase_eval:
                onsetFunction(schluter_audio_path,
                              schluter_annotations_path,
                              fn,
                              scaler_0,
                              scaler_1,
                              model_keras_cnn_0,
                              model_keras_cnn_1,
                              model_name_0,
                              model_name_1,
                              pp_threshold,
                              melbank_mth='mfcc',
                              dmfcc=False,
                              nbf=True,
                              late_fusion=False,
                              channel=1,
                              obs_cal=obs_cal)
            else:
                onsetFunctionPhrase(schluter_audio_path,
                                    schluter_annotations_path,
                                    fn,
                                    scaler_0,
                                    scaler_1,
                                    model_keras_cnn_0,
                                    model_keras_cnn_1,
                                    model_name_0,
                                    model_name_1,
                                    pp_threshold=pp_threshold,
                                    stateful=stateful,
                                    obs_cal=obs_cal)
            # except:
            #     print(fn, 'onset failed.')

    print('threshold', pp_threshold)
    recall_precision_f1_fold, recall_precision_f1_overall = eval_schluter(model_name_str)

    # TODO jingju model log path
    log_path = join(schluter_results_path,
                    weighting,
                    'schluter' + '_' +
                    filter_shape_0 +
                    phrase_str +
                    overlap_str +
                    bidi_str +
                    relu_str +
                    deep_str +
                    no_dense_str + '_' +
                    'threshold.txt')
    # log_path = join(schluter_results_path, weighting, 'schluter_jingju_model_threshold.txt')
    append_write = append_or_write(log_path)
    writeResults2TxtSchluter(log_path, append_write, pp_threshold, recall_precision_f1_overall)

    return recall_precision_f1_fold, recall_precision_f1_overall


if __name__ == '__main__':

    phrase_eval = False

    filter_shape_0 = 'jan'

    nfolds = 8

    overlap = False

    bidi = False

    relu = False

    deep = True

    no_dense = True

    overlap_str = '_overlap' if overlap else ''
    phrase_str = '_phrase' if phrase_eval else ''
    bidi_str = '_bidi_100' if bidi else ''
    relu_str = '_relu' if relu else ''
    deep_str = '_less_deep' if deep else ''
    no_dense_str = '_jingju_no_dense' if no_dense else ''

    weighting_str = 'simpleSampleWeighting' if weighting == 'simpleWeighting' else 'positiveThreeSampleWeighting'

    best_F1, best_th = 0, 0

    # first get observation and save
    pp_threshold = 0.1
    _, recall_precision_f1_overall \
        = schluterEvalSubroutine(nfolds, filter_shape_0, weighting_str, pp_threshold, obs_cal='tocal')

    if recall_precision_f1_overall[2] > best_F1:
        best_F1 = recall_precision_f1_overall[2]
        best_th = pp_threshold

    # search, load observation
    for pp_threshold in range(2,10):

        pp_threshold *= 0.1
        _, recall_precision_f1_overall \
            = schluterEvalSubroutine(nfolds, filter_shape_0, weighting_str, pp_threshold, obs_cal='toload')

        if recall_precision_f1_overall[2] > best_F1:
            best_F1 = recall_precision_f1_overall[2]
            best_th = pp_threshold

    # finer search threshold
    best_recall_precision_f1_fold = None
    best_recall_precision_f1_overall = [0,0,0]
    for pp_threshold in range(int((best_th - 0.1) * 100), int((best_th + 0.1) * 100)):

        pp_threshold *= 0.01
        recall_precision_f1_fold, recall_precision_f1_overall \
            = schluterEvalSubroutine(nfolds, filter_shape_0, weighting_str, pp_threshold, obs_cal='toload')

        if recall_precision_f1_overall[2] > best_recall_precision_f1_overall[2]:
            best_recall_precision_f1_overall = recall_precision_f1_overall
            best_recall_precision_f1_fold = recall_precision_f1_fold
            best_th = pp_threshold

    # TODO jingju model to remove
    # write recall precision f1 overall results
    txt_filename_results_schluter = 'schluter'+'_'+filter_shape_0+phrase_str+overlap_str+bidi_str+relu_str+deep_str+no_dense_str+'.txt'
    # txt_filename_results_schluter = 'schluter_jingju_model.txt'
    writeResults2TxtSchluter(join(schluter_results_path,
                                  weighting,
                                  txt_filename_results_schluter),
                             'w',
                             best_th,
                             best_recall_precision_f1_overall)

    # TODO jingju model to remove
    filename_statistical_significance = 'schluter'+'_'+filter_shape_0+phrase_str+overlap_str+bidi_str+relu_str+deep_str+no_dense_str+'.pkl'
    # filename_statistical_significance = 'schluter_jingju_model.pkl'
    # dump the statistical significance results
    pickle.dump(best_recall_precision_f1_fold,
                open(join('./statisticalSignificance/data',
                          'schluter',
                          weighting,
                          filename_statistical_significance), 'w'))
