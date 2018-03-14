"""
pitch track or chorma feature score to audio alignment and correction
"""

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
from syllablesAlignment import removeUnvoicedAndGetIdx, plotAlignment, plotAlignmentChroma, plotAlignmentPitchtrack
from stream2midi import midi2wav
from chroma_extraction import run_nnls_extraction, buildChroma, buildChroma_manualCorrectedCsv
from getScoreInfo import melodySynthesizeWithRests, melody_minus_mean
from scipy.spatial.distance import euclidean, cosine
from src.parameters import sample_number_total
from src.file_path_jingju import nacta2017_textgrid_path, nacta_textgrid_path
from src.file_path_jingju import nacta2017_wav_path, nacta_wav_path
from src.file_path_jingju import nacta2017_score_pinyin_path, nacta_score_pinyin_path
from src.file_path_jingju import nacta2017_score_pinyin_corrected_path, nacta_score_pinyin_corrected_path
from src.file_path_jingju import score2midi_path, midi2wav_path
from src.textgridParser import syllableTextgridExtraction
from src.utilFunctions import hz2cents, stringDist, pitchtrackInterp
from src.scoreParser import csvScorePinyinParser, writeCsvPinyinFromData, removePunctuation

MIN_FLOAT = 0.0000000001
SILENCE_PITCH = -12000.0

with open('./precalculated_score_json/scores.json','r') as f:
    dict_score = json.load(f)

# print(dict_score.keys())

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

    # list_score = dict_score[artist_audio_filename]

    # parse the original scores
    syllables, pinyins, syllable_durations, bpm = csvScorePinyinParser(os.path.join(score_pinyin_path, artist_name, audio_filname + '.csv'))
    lyrics_csv = [u''.join(s) for s in syllables]

    # parse the manual corrected csv scores
    syllables_manual_corrected_csv, notes, note_durations, bpm_manual_corrected_csv = csvScorePinyinParser(os.path.join('csv_corrected', artist_name, audio_filname + '.csv'))

    lyrics_manual_corrected_csv = [u''.join(syllables_mcc_line) for syllables_mcc_line in syllables_manual_corrected_csv]

    # create the corrected score path
    score_pinyin_corrected_fullpath = os.path.join(score_pinyin_corrected_path, artist_name)
    if not os.path.exists(score_pinyin_corrected_fullpath):
        os.makedirs(score_pinyin_corrected_fullpath)
    score_pinyin_corrected_filename = os.path.join(score_pinyin_corrected_fullpath, audio_filname + '.csv')

    lyrics_csv_ignore = []
    lyrics_manual_corrected_csv_ignore = []

    for i, line_list in enumerate(nestedPhonemeLists):
        line = line_list[0]
        lyrics_line = line[2]

        for ii_lmcc, lmcc_line in enumerate(lyrics_manual_corrected_csv):

            if stringDist(lyrics_csv[i], lmcc_line) > 0.5 \
                    and lyrics_csv[i] not in lyrics_csv_ignore \
                    and lmcc_line not in lyrics_manual_corrected_csv_ignore:
                lyrics_csv_ignore.append(lyrics_csv[i])
                lyrics_manual_corrected_csv_ignore.append(lmcc_line)

                syllables_line = syllables_manual_corrected_csv[ii_lmcc] #[smcc for smcc in syllables_manual_corrected_csv[ii_lmcc] if len(smcc)]
                notes_line =  notes[ii_lmcc] #[n.split('_')[0] for n in notes[ii_lmcc] if len(n)]
                note_durations_line = note_durations[ii_lmcc] #[float(nd) for nd in note_durations[ii_lmcc] if len(nd)]

                # print(notes_line)
                # print(note_durations_line)

                print(lyrics_csv[i])
                print(lmcc_line)



                # midi_filename = dict_score_line['midi_filename']
                # midi2wav(os.path.join(score2midi_path, midi_filename + '.mid',),
                #          os.path.join(midi2wav_path, midi_filename+'.wav'))
                #
                # data_midi_resyn_wav, fs_midi_resyn_wav = \
                #     sf.read(os.path.join(midi2wav_path, midi_filename+'.wav'))
                # chroma_midi_resyn, stepsize_midi_resyn = \
                #     run_nnls_extraction
                # print(syllables_line)
                # chroma_midi_resyn, idx_syllable_onset_buildChroma = buildChroma(dict_score_line, sample_number_total)
                # chroma_midi_resyn, idx_syllable_onset_buildChroma = buildChroma_manualCorrectedCsv(syllables_line, notes_line, note_durations_line, sample_number_total)
                melody_syn, idx_syllable_onset_syn = melodySynthesizeWithRests(notes_line, note_durations_line, syllables_line, sample_number_total)
                melody_syn = melody_minus_mean(melody_syn, SILENCE_PITCH)
                #
                # # remove the ending silence in chroma where the time energy is lower than threshold
                # ii_time_chroma_midi_resyn_old = chroma_midi_resyn.shape[0]
                # chroma_midi_resyn_sum = np.sum(chroma_midi_resyn, axis=1)
                # for ii_time_chroma_midi_resyn, chroma_time_energy in reversed(list(enumerate(chroma_midi_resyn_sum))):
                #     if chroma_time_energy < 0.1 and ii_time_chroma_midi_resyn == ii_time_chroma_midi_resyn_old - 1:
                #         chroma_midi_resyn = np.delete(chroma_midi_resyn, ii_time_chroma_midi_resyn, axis=0)
                #         chroma_midi_resyn_sum = np.delete(chroma_midi_resyn_sum, ii_time_chroma_midi_resyn)
                #         ii_time_chroma_midi_resyn_old = ii_time_chroma_midi_resyn
                #     else:
                #         break


                # remove zero points
                # chroma_midi_resyn[np.where(chroma_midi_resyn_sum == 0), :] = MIN_FLOAT
                # chroma_midi_resyn[] = MIN_FLOAT

                # calculate the chroma of the wav file
                start_frame = int(round(line[0] * fs_wav))
                end_frame = int(round(line[1] * fs_wav))
                data_wav_line = data_wav[start_frame:end_frame]

                # chroma_wav_line, stepsize_wav_line = \
                #         run_nnls_extraction(data_wav_line, fs_wav)
                #
                # # remove zero points
                # chroma_wav_line_sum = np.sum(chroma_wav_line, axis=1)
                # chroma_wav_line[np.where(chroma_wav_line_sum == 0), :] = MIN_FLOAT


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

                # get pitch track with the negative value rests
                data_pitch_track = vamp.collect(data_wav_line,
                                                fs_wav,
                                                "pyin:pyin",
                                                output='smoothedpitchtrack',
                                                parameters={'outputunvoiced':2})

                pitchInHz = data_pitch_track['vector'][1]
                pitchInCents = [hz2cents(p) if p>=0 else float('NaN') for p in pitchInHz]
                pitchInCents = melody_minus_mean(pitchInCents, SILENCE_PITCH)

                silence_start_onset = []
                if pitchInCents[0] > SILENCE_PITCH: silence_start_onset.append(0)
                for ii_pitchInCents in range(len(pitchInCents)-1):
                    if pitchInCents[ii_pitchInCents] == SILENCE_PITCH and pitchInCents[ii_pitchInCents+1] > SILENCE_PITCH:
                        silence_start_onset.append(ii_pitchInCents+1)
                print(silence_start_onset)

                # pitchInHz_unvoiced_removed, idx_removed2entire = removeUnvoicedAndGetIdx(pitchInHz)
                # pitch_audio_uninterp = hz2cents(pitchInHz_unvoiced_removed)

                # only interp when pitch track is longer than sample number total
                # if len(pitch_audio_uninterp) > sample_number_total:
                #     pitch_audio = pitchtrackInterp(pitch_audio_uninterp, sample_number_total)
                #     ratio_interp = len(pitch_audio)/float(len(pitch_audio_uninterp))
                # else:
                #     pitch_audio = pitch_audio_uninterp
                #     ratio_interp = 1.0
                #
                # pitch_audio -= np.mean(pitch_audio)

                # pitch_score = dict_score_line['pitchtrack_cents']
                # idx_syllable_onset_score = dict_score_line['idx_syllable_onset']
                # list_lyrics_score = dict_score_line['list_lyrics']
                # list_lyrics_score_punc_removed = [(removePunctuation(syl.encode('utf-8'))) for syl in list_lyrics_score if len(removePunctuation(syl.encode('utf-8')))]
                # syllables_punc_removed = [removePunctuation(syl) for syl in syllables[i] if len(removePunctuation(syl))]

                # _, path_dtw = fastdtw(pitch_audio, pitch_score, dist=euclidean)

                # find the best transposition
                distance, path_dtw = fastdtw(pitchInCents, melody_syn, dist=euclidean)

                print(path_dtw)

                # idx_transpose = 0
                # for ii in range(1, 12):
                #     distance_new, path_dtw_new = fastdtw(chroma_wav_line, np.roll(chroma_midi_resyn, ii , axis=1), dist=cosine)
                #
                #     if distance_new < distance:
                #         distance = distance_new
                #         path_dtw = path_dtw_new
                #         idx_transpose = ii
                #
                # if idx_transpose:
                #     chroma_midi_resyn = np.roll(chroma_midi_resyn, idx_transpose, axis = 1)

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
                    if pair[1] in idx_syllable_onset_syn and pair[1] not in idx_syllable_onset_score_ignore:
                        # get the corresponding aligned frame,
                        # compensate the shrink, map this value to the original pitch track with silence
                        if pitchInCents[pair[0]] != SILENCE_PITCH:
                            idx_corresponding_audio.append(pair[0])
                        else:
                            nearest_idx_silence_start_onset = np.argmin(np.abs(np.array(silence_start_onset)-pair[0]))
                            if silence_start_onset[nearest_idx_silence_start_onset] < pair[0] and nearest_idx_silence_start_onset+1 <= len(silence_start_onset)-1:
                                idx_corresponding_audio.append(silence_start_onset[nearest_idx_silence_start_onset+1])
                            else:
                                idx_corresponding_audio.append(silence_start_onset[nearest_idx_silence_start_onset])
                        idx_syllable_onset_score_ignore.append(pair[1])

                # plotAlignmentPitchtrack(melody_syn,
                #                         pitchInCents,
                #                         idx_syllable_onset_syn,
                #                         idx_corresponding_audio)
                # print(idx_corresponding_audio)

                syllable_dur_corrected = []
                for jj in range(1, len(idx_corresponding_audio)):
                    syllable_dur = idx_corresponding_audio[jj] - idx_corresponding_audio[jj-1]
                    syllable_dur_corrected.append(syllable_dur)
                syllable_dur_corrected.append(len(pitchInCents)-idx_corresponding_audio[-1])

                # print(syllable_dur_corrected)

                # replace 0 in list by the minimum non 0 value
                if 0 in syllable_dur_corrected:
                    value2replace = min([sdr for sdr in syllable_dur_corrected if sdr != 0])
                    syllable_dur_corrected = [value2replace if sdr ==0 else sdr for sdr in syllable_dur_corrected ]
                # print(list_lyrics_score, len(list_lyrics_score))

                syllable_durations[i] = syllable_dur_corrected

                # print(syllable_dur_corrected)


    writeCsvPinyinFromData(score_pinyin_corrected_filename,
                           syllables=syllables,
                           pinyins=pinyins,
                           syllable_durations=syllable_durations,
                           bpm=bpm)