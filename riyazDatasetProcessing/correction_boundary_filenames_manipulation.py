import os
import csv
import json
import shutil
from operator import itemgetter
from filepath_riyaz import *
from labWriter import boundaryLabWriter
from scoreParser import writeCsv

def correction_boundary_filenames_parser(filename_boundary_correction_csv):
    """
    parse the filename in correction boundary .csv
    :return: filename list
    """
    list_filename = []
    with open(filename_boundary_correction_csv, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            for fn in row:
                if len(fn) and '.mp3' in fn:
                    list_filename.append(fn.replace('.mp3', ''))
    return list_filename

def mp3JsonRefFinder(path_riyaz_dataset, filename):
    """
    find the mp3, json and Reference score paths for filename in Riyaz dataset
    :param path_riyaz_dataset:
    :param filename:
    :return:
    """
    for root, dirs, files in os.walk(path_riyaz_dataset):
        for fn in files:
            if filename in fn:
                filename_mp3 = os.path.join(root, filename+'.mp3')
                filename_json = os.path.join(root, filename + '.json')
                filename_ref = os.path.join(root, 'reference.trans')
                return filename_mp3, filename_json, filename_ref

def startEndTime(filename_json):
    """
    collect start and end time in filename json
    :param filename_json:
    :return:
    """
    list_start_end_time = []
    data = json.load(open(filename_json))
    intonation = data['IntonationRating']
    for i in intonation:
        start_end_time = [i['start_time'], i['end_time']]
        if start_end_time not in list_start_end_time:
            list_start_end_time.append(start_end_time)

    # sort the list according to the first element
    list_start_end_time = sorted(list_start_end_time, key=itemgetter(0))
    return list_start_end_time

def keepOnsetTime(list_start_end_time):
    """
    replace the offset time by the onset time of the subsequent element
    :param list_start_end_time:
    :return:
    """
    for ii in range(len(list_start_end_time)-1):
        list_start_end_time[ii][1] = list_start_end_time[ii+1][0]
    return list_start_end_time

def dumpGroundtruth(filename_json):
    """
    dump a syllable boundary groundtruth dictionary from the json
    :param filename_json:
    :return:
    """
    list_start_end_time = startEndTime(filename_json)
    bn = os.path.basename(filename_json)
    return {bn.replace('.json', ''): list_start_end_time}, len(list_start_end_time)

def readRef(filename_ref):
    """
    read the reference file into a list
    :param filename_ref:
    :return:
    """
    list_ref = []
    with open(filename_ref, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            list_ref.append(row)
    return list_ref

def reformScoreElements(list_ref):
    """
    reform score elements into lists
    :param list_ref:
    :return:
    """
    syllables = []
    syllable_durations = []
    bpm = []
    for r in list_ref:
        syllables.append(r[3])
        syllable_durations.append(float(r[1])-float(r[0]))
        bpm.append(60)
    syllables = [syllables]
    syllable_durations = [syllable_durations]
    return syllables, syllable_durations, bpm

def dumpRef(filename_json, filename_ref):
    """
    dump the score into a dictionary
    :param filename_json:
    :param filename_ref:
    :return:
    """
    list_ref = readRef(filename_ref)
    bn = os.path.basename(filename_json)
    return {bn.replace('.json', ''): list_ref}, len(list_ref)

if __name__ == '__main__':
    list_filename = correction_boundary_filenames_parser(filename_boundary_correction_csv)

    for fn in list_filename:
        filename_mp3, filename_json, filename_ref = mp3JsonRefFinder(filepath_riyaz_dataset, fn)
        print(filename_mp3, filename_json, filename_ref)
        list_gt = startEndTime(filename_json)
        list_gt = keepOnsetTime(list_gt)
        list_s = readRef(filename_ref)
        syllables, syllable_durations, bpm = reformScoreElements(list_s)
        shutil.copy2(filename_mp3, os.path.join('./mp3'))
        boundaryLabWriter(list_gt, os.path.join('./groundtruth', fn+'.lab'), label=False)
        writeCsv(os.path.join('./score', fn+'.csv'), syllables, syllable_durations, bpm)