import numpy as np
from os.path import join
from src.labParser import lab2WordList
from src.scoreParser import csvDurationScoreParser
from src.scoreParser import csvScorePinyinParser
from src.textgridParser import textGrid2WordList
from src.textgridParser import wordListsParseByLines
from src.labWriter import onsetLabWriter


def data_parser(artist_path,
                wav_path,
                textgrid_path,
                rn,
                score_file,
                lab):
    """parse the wav filename, text grid and score"""

    if not lab:
        # ground truth text grid
        ground_truth_text_grid_file = join(textgrid_path, artist_path, rn + '.TextGrid')

        # wav
        wav_file = join(wav_path, artist_path, rn + '.wav')

        # parse line
        line_list = textGrid2WordList(ground_truth_text_grid_file, whichTier='line')

        # parse syllable
        syllable_list = textGrid2WordList(ground_truth_text_grid_file, whichTier='dianSilence')

        # parse lines of ground truth
        nested_syllable_lists, _, _ = wordListsParseByLines(line_list, syllable_list)

        # parse score
        syllables, pinyins, syllable_durations, bpm = csvScorePinyinParser(score_file)
    else:
        ground_truth_text_grid_file = join(textgrid_path, artist_path, rn + '.lab')
        wav_file = join(wav_path, artist_path, rn + '.mp3')
        line_list = [lab2WordList(ground_truth_text_grid_file, label=True)]
        syllables, syllable_durations, bpm = csvDurationScoreParser(score_file)
        nested_syllable_lists = None
        pinyins = None

    return nested_syllable_lists, wav_file, line_list, syllables, syllable_durations, bpm, pinyins


def get_line_properties(lab, line, hopsize_t):
    """get singing line properties,
    length, lyrics, starting frame, ending frame"""
    if not lab:
        time_line = line[1] - line[0]
        lyrics_line = line[2]
        print('Line lyrics:', lyrics_line)

        frame_start = int(round(line[0] / hopsize_t))
        frame_end = int(round(line[1] / hopsize_t))
    else:
        time_line = line[-1][1] - line[0][0]
        lyrics_line = None

        frame_start = int(round(line[0][0] / hopsize_t))
        frame_end = int(round(line[-1][1] / hopsize_t))

    return time_line, lyrics_line, frame_start, frame_end


def boundary_decoding(decoding_method,
                      obs_i,
                      duration_score,
                      varin,
                      threshold,
                      hopsize_t,
                      viterbiDecoding,
                      OnsetPeakPickingProcessor):

    """decode boundary"""
    # decoding: Viterbi or peak picking
    if decoding_method == 'viterbi':
        # segmental decoding
        obs_i[0] = 1.0
        obs_i[-1] = 1.0
        i_boundary = viterbiDecoding.viterbiSegmental2(obs_i, duration_score, varin)
        label = True
    else:
        arg_pp = {'threshold': threshold,
                  'smooth': 0,
                  'fps': 1. / hopsize_t,
                  'pre_max': hopsize_t,
                  'post_max': hopsize_t}

        peak_picking = OnsetPeakPickingProcessor(**arg_pp)
        i_boundary = peak_picking.process(obs_i)
        i_boundary = np.append(i_boundary, (len(obs_i) - 1) * hopsize_t)
        i_boundary /= hopsize_t
        label = False
    return i_boundary, label


def get_results_decoding_path(decoding_method,
                              bool_corrected_score_duration,
                              eval_results_path):
    """get results decoding path"""
    if decoding_method == 'viterbi':
        # segmental decoding
        # corrected score duration is used in experiment
        # the score duration is corrected by using audio-to-score alignment
        if bool_corrected_score_duration:
            eval_results_decoding_path = eval_results_path + '_corrected_score_duration'
        else:
            eval_results_decoding_path = eval_results_path
    else:
        eval_results_decoding_path = eval_results_path + '_peakPickingMadmom'
    return eval_results_decoding_path


def get_boundary_list(lab,
                      decoding_method,
                      time_boundary_start,
                      time_boundary_end,
                      pinyins,
                      syllables,
                      i_line):
    """get the correct boundary list to output"""
    # write boundary lab file
    if not lab:
        if decoding_method == 'viterbi':
            boundary_list = zip(time_boundary_start.tolist(), time_boundary_end.tolist(),
                                filter(None, pinyins[i_line]))
        else:
            boundary_list = zip(time_boundary_start.tolist(), time_boundary_end.tolist())
    else:
        if decoding_method == 'viterbi':
            boundary_list = zip(time_boundary_start.tolist(), time_boundary_end.tolist(), syllables[i_line])
        else:
            boundary_list = zip(time_boundary_start.tolist(), time_boundary_end.tolist())
    return boundary_list


def write_results_2_txt_jingju(filename,
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


def write_results_2_txt_schluter(filename,
                                 append_write,
                                 best_th,
                                 recall_precision_f1_overall):

    """
    write the schluter evaluation results to text
    :param filename:
    :param append_write:
    :param best_th:
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


def wav_annotation_loader_parser(audio_path,
                                 annotation_path,
                                 filename,
                                 annotationCvParser):
    # load annotation
    annotation_filename = join(annotation_path, filename + '.onsets')

    # load audio
    audio_filename = join(audio_path, filename + '.flac')

    # parse annotation
    groundtruth_onset = annotationCvParser(annotation_filename)
    groundtruth_onset = [float(gto) for gto in groundtruth_onset]

    return audio_filename, groundtruth_onset


def peak_picking_detected_onset_saver_schluter(pp_threshold,
                                               obs_i,
                                               model_name_0,
                                               model_name_1,
                                               filename,
                                               hopsize_t,
                                               OnsetPeakPickingProcessor,
                                               eval_results_path):

    # madmom peak picking
    arg_pp = {'threshold': pp_threshold,
              'smooth': 0.05,
              'fps': 1. / hopsize_t,
              'pre_max': hopsize_t,
              'post_max': hopsize_t,
              'combine': 0}
    peak_picking = OnsetPeakPickingProcessor(**arg_pp)
    detected_onsets = peak_picking.process(obs_i)

    # save detected onsets
    filename_syll_lab = join(eval_results_path, model_name_0 + model_name_1, filename + '.syll.lab')

    onsetLabWriter(detected_onsets, filename_syll_lab)

    return detected_onsets