from os import walk, listdir, mkdir
from praatio import tgio
from os.path import isfile, join, exists
from src.file_path_jingju_no_rnn import *
from src.labParser import lab2WordList

def textgridSyllableSegmentation(wav_path,
                                textgrid_path,
                                ):

    list_file_path_name = []
    for file_path_name in walk(textgrid_path):
        list_file_path_name.append(file_path_name)

    list_artist_level_path = list_file_path_name[0][1]

    for artist_path in list_artist_level_path:
        textgrid_artist_path = join(textgrid_path, artist_path)
        recording_names = [f for f in listdir(textgrid_artist_path) if isfile(join(textgrid_artist_path, f))]
        # print(recording_names)

        for rn in recording_names:
            print rn
            rn = rn.split('.')[0]
            if len(rn) == 0:
                continue

            groundtruth_textgrid_file   = join(textgrid_path, artist_path, rn+'.TextGrid')

            tg = tgio.openTextgrid(groundtruth_textgrid_file)

            list_line = tg.tierDict['line']
            list_dianSilence = []
            for i_line, (start, stop, label) in enumerate(list_line.entryList):
                syll_lab_file = join('./output', artist_path, rn+'_'+str(i_line+1)+'.syll.lab')
                # do the line-level segmentation
                try:
                    lab_list = lab2WordList(syll_lab_file)
                    # print lab_list
                    lab_list_time_corrected = [[lab_cell[0]+start, lab_cell[1]+start, lab_cell[2]] for lab_cell in lab_list if lab_cell[0] != lab_cell[1]]
                    lab_list_time_corrected[-1][1] = stop
                    list_dianSilence += lab_list_time_corrected
                except:
                    continue

            # add dianSilence tier to line tier
            dianSilenceTier = tgio.IntervalTier('dianSilence', list_dianSilence, 0, tg.maxTimestamp)
            tg.addTier(dianSilenceTier)

            output_dianSilence_path = join('./output_dianSilence', artist_path)
            if not exists(output_dianSilence_path):
                mkdir(output_dianSilence_path)

            tg.save(join(output_dianSilence_path, rn+'.TextGrid'))


textgridSyllableSegmentation(wav_path, textgrid_path)