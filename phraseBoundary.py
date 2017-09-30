import labWriter
import os
from src.filePath import *
from trainingSampleCollection import getTestTrainRecordingsMaleFemale, getTestTrainrecordingsRiyaz

# testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordings()
#
# for artist_path, recording_name in testNacta2017:
#
#     textgrid_file       = os.path.join(nacta2017_textgrid_path, artist_path, recording_name+'.TextGrid')
#     boundary_lab_file   = os.path.join(nacta2017_segPhrase_path, artist_path,  recording_name+'.lab')
#
#     labWriter.phraseBoundaryWriter(textgrid_file, boundary_lab_file)
#
# for artist_path, recording_name in testNacta:
#
#     textgrid_file       = os.path.join(nacta_textgrid_path, artist_path, recording_name+'.TextGrid')
#     boundary_lab_file   = os.path.join(nacta_segPhrase_path, artist_path,  recording_name+'.lab')
#
#     labWriter.phraseBoundaryWriter(textgrid_file, boundary_lab_file)
#
testRiyaz, trainRiyaz = getTestTrainrecordingsRiyaz()

for artist_path, recording_name in testRiyaz:

    textgrid_file = os.path.join(riyaz_groundtruthlab_path, artist_path, recording_name + '.lab')
    boundary_lab_file = os.path.join(riyaz_segPhrase_path, artist_path, recording_name + '.lab')

    labWriter.phraseBoundaryWriterLab(textgrid_file, boundary_lab_file)