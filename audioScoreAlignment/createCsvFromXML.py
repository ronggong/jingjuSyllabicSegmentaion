#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from src.scoreParser import writeCsvPinyinFromData



with open('./precalculated_score_json/scores.json','r') as f:
    dict_score = json.load(f)


for key in dict_score:
    foldername, filename = key.split(' ')

    score_xml_fullpath = os.path.join('csv', foldername)

    if not os.path.exists(score_xml_fullpath):
        os.makedirs(score_xml_fullpath)

    score_xml_csv_filename = os.path.join(score_xml_fullpath, filename + '.csv')

    syllables = []
    note_durations = []
    notes = []

    for l in dict_score[key]:
        syllables_phrase = []
        note_durations_phrase = []
        notes_phrase = []
        for note in l['notesWithRealRests']:

            if note['noteName']:
                notes_phrase.append(note['noteName']+'_'+str(note['freq']))
            else:
                notes_phrase.append('')

            if note['lyric']:
                syllables_phrase.append(note['lyric'].encode("utf8"))
            else:
                syllables_phrase.append('')

            note_durations_phrase.append(note['quarterLength'])

        syllables.append(syllables_phrase)
        notes.append(notes_phrase)
        note_durations.append(note_durations_phrase)
    bpm = [60]*len(syllables)

    writeCsvPinyinFromData(score_xml_csv_filename,
                           syllables=syllables,
                           pinyins=notes,
                           syllable_durations=note_durations,
                           bpm=bpm)