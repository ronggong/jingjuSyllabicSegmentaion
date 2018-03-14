# -*- coding: utf-8 -*-

from jingjuScores import getMelodicLine
import music21
from src.file_path_jingju import *
from src.utilFunctions import hz2cents,pitchtrackInterp
from stream2midi import stream2midi
from math import isnan
import numpy as np
from os import path
import csv,json

def getDictScoreInfo(score_info_filepath):
    dict_score_info = {}
    with open(score_info_filepath, 'r') as csvfile:
        score_info = csv.reader(csvfile,delimiter=",")

        score_filename_old = None
        audio_filename_old = None

        line_number = 0
        for row in score_info:
            lyrics_filename = row[0]
            audio_filename = row[1]
            if len(lyrics_filename) and len(audio_filename):
                line_number = 0
                score_filename_old = lyrics_filename
                audio_filename_old = audio_filename
                # print(audio_filename_old)
                dict_score_info[audio_filename_old] = []
            else:
                line_number += 1

            try:
                dict_score_info[audio_filename_old].append({'line_number': line_number,
                                                            'score_filename':score_filename_old,
                                                            'lyrics': row[2],
                                                            'start_offset': float(row[3]),
                                                            'end_offset': float(row[4])})
            except ValueError:
                print(score_filename_old, str(line_number), 'valueError: ', row[3],  row[4])

        return dict_score_info

def getScores(full_score_filename, dict_score_line):
    """
    get notes by giving the start and end offsets
    :param full_score_filename:
    :param key:
    :param dict_score_line:
    :return:
    """
    #TODO add freq to notesWithRealRests
    start = dict_score_line['start_offset']
    end = dict_score_line['end_offset']

    line = getMelodicLine(full_score_filename, start, end, show=False)

    notes = []
    notesWithRests = [] # extend the note duration with the rests
    notesWithRealRests = []
    # for note in line.flat.notes.stream():

    # get both notes and rests
    line_stream = line.flat.notesAndRests.stream()
    midi_stream = music21.stream.Stream()
    mm1 = music21.tempo.MetronomeMark(number=60)
    midi_stream.append(mm1)
    for ii_note, note in enumerate(line_stream):
        if note.quarterLength != 0:
            midi_stream.append(note)
            if not note.isRest:
                dict_note_info = {'freq':note.pitch.freq440,
                                  'lyric':note.lyric,
                                  'quarterLength':float(note.quarterLength)}
                notes.append(dict_note_info.copy())
                notesWithRests.append(dict_note_info.copy())
                notesWithRealRests.append({'freq':note.pitch.freq440,
                                           'noteName':note.name,
                                           'lyric': note.lyric,
                                           'quarterLength':float(note.quarterLength)})
            else:
                if ii_note != len(line_stream)-1 and len(notesWithRests):
                    # if it's a rest, then append the rest length to the last note
                    # print(line_stream[0].quarterLength, line_stream[0].lyric, line_stream[0].isRest)
                    notesWithRests[-1]['quarterLength'] += note.quarterLength
                    notesWithRealRests.append({'freq':None,
                                               'noteName':None,
                                               'lyric': note.lyric,
                                               'quarterLength':float(note.quarterLength)})

    # append a rest to mute the last note tail
    fake_note = music21.note.Note()
    fake_note.volume.velocity = 0

    midi_stream.append(fake_note)

    dict_score_line['notes'] = notes
    dict_score_line['notesWithRests'] = notesWithRests
    dict_score_line['notesWithRealRests'] = notesWithRealRests

    return dict_score_line, midi_stream

def convertScore2midi(full_score_filename, full_midi_filename, midi_stream):
    """
    convert music 21 stream line into midi
    :param full_score_filename:
    :param full_midi_filename: saved midi path
    :param midi_stream:
    :return:
    """

    stream2midi(midi_stream, full_midi_filename)

def melodySynthesize(notes_pitch_hz, notes_quarterlength, notes_lyric, notesWithRests_quarterlength, sample_number_total):
    '''
    :param notes_quarterlength: a list of the note quarterLength
    :return: list, pitch track values
    '''
    notes_pitch_cents = hz2cents(np.array(notes_pitch_hz))

    # print(notes_pitch_cents)
    # print(notes_quarterlength)

    length_total = sum(notes_quarterlength)
    length_total_withRests = sum(notesWithRests_quarterlength)

    melody = []
    melody_withRests =[]
    idx_syllable_onset = []
    idx_syllable_onset_withRests = []
    list_lyrics = []
    for ii in range(len(notes_quarterlength)):
        sample_note = int(round((notes_quarterlength[ii]/length_total)*sample_number_total))
        sample_note_withRest = int(round((notesWithRests_quarterlength[ii]/length_total_withRests)*sample_number_total))

        if sample_note != 0:
            # if note contains a lyric, add the onset index
            if notes_lyric[ii]:
                idx_syllable_onset.append(len(melody))
                idx_syllable_onset_withRests.append(len(melody_withRests))
                list_lyrics.append(notes_lyric[ii])
            melody += [notes_pitch_cents[ii]]*sample_note
            melody_withRests += [notes_pitch_cents[ii]]*sample_note_withRest

    melody = np.array(melody)

    idx_syllable_onset = [int(round(iso*(float(sample_number_total)/len(melody)))) for iso in idx_syllable_onset]

    # interpolation
    if len(melody) != sample_number_total:
        melody = pitchtrackInterp(melody, sample_number_total)

    melody = melody - np.mean(melody)

    return melody.tolist(), melody_withRests, idx_syllable_onset, idx_syllable_onset_withRests, list_lyrics

def melodySynthesizeWithRests(notes_pitch_hz, notes_quarterlength, notes_lyric, sample_number_total):
    '''
    :param notes_quarterlength: a list of the note quarterLength
    :return: list, pitch track values
    '''
    notes_pitch_cents = [hz2cents(float(n.split('_')[1])) if len(n) else float('NaN') for n in notes_pitch_hz]

    # print(notes_pitch_cents)
    # print(notes_quarterlength)

    length_total = sum([float(nd) for nd in notes_quarterlength if len(nd)])

    melody = []
    idx_syllable_onset = []
    for ii in range(len(notes_quarterlength)):
        if len(notes_quarterlength[ii]):
            sample_note = int(round((float(notes_quarterlength[ii])/length_total)*sample_number_total))
            if sample_note != 0:
                # if note contains a lyric, add the onset index
                if notes_lyric[ii]:
                    idx_syllable_onset.append(len(melody))
                melody += [notes_pitch_cents[ii]]*sample_note

    # melody = np.array(melody)
    #
    # idx_syllable_onset = [int(round(iso*(float(sample_number_total)/len(melody)))) for iso in idx_syllable_onset]
    #
    # # interpolation
    # if len(melody) != sample_number_total:
    #     melody = pitchtrackInterp(melody, sample_number_total)

    return melody, idx_syllable_onset

def melody_minus_mean(melody, silence_pitch):
    mean_melody = np.mean([m for m in melody if not isnan(m)])
    melody = [m - mean_melody if not isnan(m) else silence_pitch for m in melody]
    return melody

##-- dump json scores
if __name__ == '__main__':

    # score information *.csv
    from src.parameters import sample_number_total
    import uuid

    score_info_filepath = path.join(score_path, score_info_filename)
    dict_score_infos = getDictScoreInfo(score_info_filepath)

    for audio_filename in dict_score_infos:
        list_score = dict_score_infos[audio_filename]
        for dict_score_line in list_score:
            score_filename = dict_score_line['score_filename']
            full_score_filename = join(score_path, score_filename)
            dict_score_line, midi_stream = getScores(full_score_filename, dict_score_line)

            # synthesize melody
            notes_quarterLength = []
            notesWithRests_quarterLength = []
            notes_pitch_hz = []
            notes_lyric = []
            for dict_note in dict_score_line['notes']:
                notes_pitch_hz.append(dict_note['freq'])
                notes_quarterLength.append(dict_note['quarterLength'])
                notes_lyric.append(dict_note['lyric'])

            for dict_noteWithRest in dict_score_line['notesWithRests']:
                notesWithRests_quarterLength.append(dict_noteWithRest['quarterLength'])


            pitchtrack_cents, \
            pitchtrack_hz_withRests, \
            idx_syllable_onset, \
            idx_syllable_onset_withRests, \
            list_lyrics = melodySynthesize(notes_pitch_hz,
                                                                        notes_quarterLength,
                                                                        notes_lyric,
                                                                        notesWithRests_quarterLength,
                                                                        sample_number_total)
            dict_score_line['pitchtrack_cents'] = pitchtrack_cents
            dict_score_line['pitchtrack_hz_withRests'] = pitchtrack_hz_withRests
            dict_score_line['idx_syllable_onset'] = idx_syllable_onset
            dict_score_line['idx_syllable_onset_withRests'] = idx_syllable_onset_withRests
            dict_score_line['list_lyrics'] = list_lyrics

            midi_filename = str(uuid.uuid4())
            midi_score_line_full_filename = join(score2midi_path, midi_filename + '.mid')
            convertScore2midi(full_score_filename, midi_score_line_full_filename, midi_stream)
            dict_score_line['midi_filename'] = midi_filename

    # print dict_score_infos[key]
    with open('./precalculated_score_json/scores.json','w') as outfile:
        json.dump(dict_score_infos,outfile)

    # with open('scores.json','r') as f:
    #     dict_score_infos = json.load(f)
