#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import soundfile as sf
import os
import vamp
import numpy as np
from fastdtw import fastdtw
import nwalign as nw
from syllablesAlignment import convertSyl2Letters, adjustSyllableDurByAlignment
from syllablesAlignment import removeUnvoicedAndGetIdx, plotAlignment, plotAlignmentChroma
from stream2midi import midi2wav
from chroma_extraction import run_nnls_extraction, buildChroma
from scipy.spatial.distance import euclidean, cosine
from src.parameters import sample_number_total
from src.filePath import nacta2017_textgrid_path, nacta_textgrid_path
from src.filePath import nacta2017_wav_path, nacta_wav_path
from src.filePath import nacta2017_score_pinyin_path, nacta_score_pinyin_path
from src.filePath import nacta2017_score_pinyin_corrected_path, nacta_score_pinyin_corrected_path
from src.filePath import score2midi_path, midi2wav_path
from src.textgridParser import syllableTextgridExtraction
from src.utilFunctions import hz2cents, stringDist, pitchtrackInterp
from src.scoreParser import csvScorePinyinParser, writeCsvPinyinFromData, removePunctuation

MIN_FLOAT = 0.0000000001

with open('./precalculated_score_json/scores.json','r') as f:
    dict_score = json.load(f)

for artist_audio_filename in dict_score:

    # print(artist_audio_filename)

    artist_name, audio_filname = artist_audio_filename.split(' ')

    # get the correct textgrid path
    if artist_name.lower() == 'danall' or artist_name.lower() == 'laosheng':
        textgrid_path = nacta_textgrid_path
        wav_path = nacta_wav_path
        score_pinyin_path = nacta_score_pinyin_path
        score_pinyin_corrected_path = nacta_score_pinyin_corrected_path
    else:
        textgrid_path = nacta2017_textgrid_path
        wav_path = nacta2017_wav_path
        score_pinyin_path = nacta2017_score_pinyin_path
        score_pinyin_corrected_path = nacta2017_score_pinyin_corrected_path

    nestedPhonemeLists, _, _ = syllableTextgridExtraction(os.path.join(textgrid_path, artist_name),
                                                          audio_filname,
                                                          'line',
                                                          'dianSilence')
    data_wav, fs_wav = sf.read(os.path.join(wav_path, artist_name, audio_filname + '.wav'))
    if len(data_wav.shape) == 2:
        data_wav = data_wav[:,0]

    list_score = dict_score[artist_audio_filename]

    syllables, pinyins, syllable_durations, bpm = csvScorePinyinParser(os.path.join(score_pinyin_path, artist_name, audio_filname + '.csv'))

    score_pinyin_corrected_fullpath = os.path.join(score_pinyin_corrected_path, artist_name)
    if not os.path.exists(score_pinyin_corrected_fullpath):
        os.makedirs(score_pinyin_corrected_fullpath)
    score_pinyin_corrected_filename = os.path.join(score_pinyin_corrected_fullpath, audio_filname + '.csv')

    for i, line_list in enumerate(nestedPhonemeLists):
        line = line_list[0]
        lyrics_line = line[2]

        for dict_score_line in list_score:

            if stringDist(lyrics_line, dict_score_line['lyrics']) > 0.5:
                # print(lyrics_line)
                # print(dict_score_line['lyrics'])



                # midi_filename = dict_score_line['midi_filename']
                # midi2wav(os.path.join(score2midi_path, midi_filename + '.mid',),
                #          os.path.join(midi2wav_path, midi_filename+'.wav'))
                #
                # data_midi_resyn_wav, fs_midi_resyn_wav = \
                #     sf.read(os.path.join(midi2wav_path, midi_filename+'.wav'))
                # chroma_midi_resyn, stepsize_midi_resyn = \
                #     run_nnls_extraction
                chroma_midi_resyn, idx_syllable_onset_buildChroma = buildChroma(dict_score_line, sample_number_total)

                #
                # # remove the ending silence in chroma where the time energy is lower than threshold
                # ii_time_chroma_midi_resyn_old = chroma_midi_resyn.shape[0]
                chroma_midi_resyn_sum = np.sum(chroma_midi_resyn, axis=1)
                # for ii_time_chroma_midi_resyn, chroma_time_energy in reversed(list(enumerate(chroma_midi_resyn_sum))):
                #     if chroma_time_energy < 0.1 and ii_time_chroma_midi_resyn == ii_time_chroma_midi_resyn_old - 1:
                #         chroma_midi_resyn = np.delete(chroma_midi_resyn, ii_time_chroma_midi_resyn, axis=0)
                #         chroma_midi_resyn_sum = np.delete(chroma_midi_resyn_sum, ii_time_chroma_midi_resyn)
                #         ii_time_chroma_midi_resyn_old = ii_time_chroma_midi_resyn
                #     else:
                #         break


                # remove zero points
                chroma_midi_resyn[np.where(chroma_midi_resyn_sum == 0), :] = MIN_FLOAT
                # chroma_midi_resyn[] = MIN_FLOAT

                # calculate the chroma of the wav file
                start_frame = int(round(line[0] * fs_wav))
                end_frame = int(round(line[1] * fs_wav))
                data_wav_line = data_wav[start_frame:end_frame]
                chroma_wav_line, stepsize_wav_line = \
                        run_nnls_extraction(data_wav_line, fs_wav)

                # remove zero points
                chroma_wav_line_sum = np.sum(chroma_wav_line, axis=1)
                chroma_wav_line[np.where(chroma_wav_line_sum == 0), :] = MIN_FLOAT


                # idx_syllable_onset_score_withRests = dict_score_line['idx_syllable_onset_withRests']
                # ratio_chroma_pitchtrack = chroma_midi_resyn.shape[0]/float(len(dict_score_line['pitchtrack_hz_withRests']))
                # chroma_syllable_onset_midi_resyn = np.array(idx_syllable_onset_score_withRests) * ratio_chroma_pitchtrack
                # chroma_syllable_onset_midi_resyn = np.around(chroma_syllable_onset_midi_resyn, decimals=0).astype(int)


                # import matplotlib.pyplot as plt
                # plt.subplot(2, 1, 1)
                # plt.imshow(np.transpose(chroma_midi_resyn))
                # plt.title('reference up, student bottom')
                # plt.subplot(2, 1, 2)
                # plt.imshow(np.transpose(chroma_wav_line))
                # plt.show()
    #
    #             # get pitch track with the negative value rests
    #             data_pitch_track = vamp.collect(data_wav_line,
    #                                             fs_wav,
    #                                             "pyin:pyin",
    #                                             output='smoothedpitchtrack',
    #                                             parameters={'outputunvoiced':2})
    #
    #             pitchInHz = data_pitch_track['vector'][1]
    #             pitchInHz_unvoiced_removed, idx_removed2entire = removeUnvoicedAndGetIdx(pitchInHz)
    #             pitch_audio_uninterp = hz2cents(pitchInHz_unvoiced_removed)
    #
    #             # only interp when pitch track is longer than sample number total
    #             if len(pitch_audio_uninterp) > sample_number_total:
    #                 pitch_audio = pitchtrackInterp(pitch_audio_uninterp, sample_number_total)
    #                 ratio_interp = len(pitch_audio)/float(len(pitch_audio_uninterp))
    #             else:
    #                 pitch_audio = pitch_audio_uninterp
    #                 ratio_interp = 1.0
    #
    #             pitch_audio -= np.mean(pitch_audio)
    #
                # pitch_score = dict_score_line['pitchtrack_cents']
                # idx_syllable_onset_score = dict_score_line['idx_syllable_onset']
                list_lyrics_score = dict_score_line['list_lyrics']
                list_lyrics_score_punc_removed = [(removePunctuation(syl.encode('utf-8'))) for syl in list_lyrics_score if len(removePunctuation(syl.encode('utf-8')))]
                syllables_punc_removed = [removePunctuation(syl) for syl in syllables[i] if len(removePunctuation(syl))]

                # _, path_dtw = fastdtw(pitch_audio, pitch_score, dist=euclidean)

                # find the best transposition
                distance, path_dtw = fastdtw(chroma_wav_line, chroma_midi_resyn, dist=cosine)
                idx_transpose = 0
                for ii in range(1, 12):
                    distance_new, path_dtw_new = fastdtw(chroma_wav_line, np.roll(chroma_midi_resyn, ii , axis=1), dist=cosine)

                    if distance_new < distance:
                        distance = distance_new
                        path_dtw = path_dtw_new
                        idx_transpose = ii

                if idx_transpose:
                    chroma_midi_resyn = np.roll(chroma_midi_resyn, idx_transpose, axis = 1)

                # plotAlignmentChroma(np.transpose(chroma_midi_resyn),
                #                     np.transpose(chroma_wav_line),
                #                     path_dtw,
                #                     distance,
                #                     idx_syllable_onset_buildChroma)


                # print(path_dtw)

                # plotAlignment(pitch_score,
                #               hz2cents(pitchInHz),
                #               path_dtw,
                #               idx_syllable_onset_score,
                #               idx_removed2entire,
                #               ratio_interp)

                # # find corresponding syllable index in audio pitch track
                # idx_corresponding_audio = []
                # idx_syllable_onset_score_ignore = []
                # for pair in path_dtw:
                #     if pair[1] in idx_syllable_onset_score and pair[1] not in idx_syllable_onset_score_ignore:
                #         # get the corresponding aligned frame,
                #         # compensate the shrink, map this value to the original pitch track with silence
                #         idx_corresponding_audio.append(idx_removed2entire[str(int(pair[0]/ratio_interp))])
                #         idx_syllable_onset_score_ignore.append(pair[1])
                # # print(idx_corresponding_audio, len(pitch_audio))

                # find corresponding syllable index in audio pitch track
                idx_corresponding_audio = []
                idx_syllable_onset_score_ignore = []
                for pair in path_dtw:
                    if pair[1] in idx_syllable_onset_buildChroma and pair[1] not in idx_syllable_onset_score_ignore:
                        # get the corresponding aligned frame,
                        # compensate the shrink, map this value to the original pitch track with silence
                        idx_corresponding_audio.append(pair[0])
                        idx_syllable_onset_score_ignore.append(pair[1])
                print(idx_corresponding_audio)

                syllable_dur_corrected = []
                for jj in range(1, len(idx_corresponding_audio)):
                    syllable_dur = idx_corresponding_audio[jj] - idx_corresponding_audio[jj-1]
                    syllable_dur_corrected.append(syllable_dur)
                syllable_dur_corrected.append(chroma_wav_line.shape[0]-idx_corresponding_audio[-1])

                # replace 0 in list by the minimum non 0 value
                if 0 in syllable_dur_corrected:
                    value2replace = min([sdr for sdr in syllable_dur_corrected if sdr != 0])
                    syllable_dur_corrected = [value2replace if sdr ==0 else sdr for sdr in syllable_dur_corrected ]
                # print(list_lyrics_score, len(list_lyrics_score))


                # align two syllable lists since sometimes they are not the same
                # print(list_lyrics_score_punc_removed, syllables_punc_removed)
                lyrics_score_letter, syllables_letter, dict_letter2syl = \
                    convertSyl2Letters(list_lyrics_score_punc_removed, syllables_punc_removed)
                lyrics_score_letter_aligned, syllables_letter_aligned = \
                    nw.global_align(lyrics_score_letter, syllables_letter)
                print(lyrics_score_letter_aligned, syllables_letter_aligned)

                if lyrics_score_letter_aligned != syllables_letter_aligned:
                    syllable_dur_corrected = adjustSyllableDurByAlignment(durations=syllable_dur_corrected,
                                                                          syllables_score_aligned=lyrics_score_letter_aligned,
                                                                          syllables_ground_truth_aligned=syllables_letter_aligned)

                syllable_durations[i] = syllable_dur_corrected

                # print(syllable_dur_corrected)


    writeCsvPinyinFromData(score_pinyin_corrected_filename,
                           syllables=syllables,
                           pinyins=pinyins,
                           syllable_durations=syllable_durations,
                           bpm=bpm)