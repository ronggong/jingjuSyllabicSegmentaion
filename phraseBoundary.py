import labWriter
import os
from src.filePath import *
from trainingSampleCollection import getTestTrainRecordings

testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordings()

for artist_path, recording_name in testNacta2017:

    textgrid_file       = os.path.join(nacta2017_textgrid_path, artist_path, recording_name+'.TextGrid')
    boundary_lab_file   = os.path.join(nacta2017_segPhrase_path, artist_path,  recording_name+'.lab')

    labWriter.phraseBoundaryWriter(textgrid_file, boundary_lab_file)

for artist_path, recording_name in testNacta:

    textgrid_file       = os.path.join(nacta_textgrid_path, artist_path, recording_name+'.TextGrid')
    boundary_lab_file   = os.path.join(nacta_segPhrase_path, artist_path,  recording_name+'.lab')

    labWriter.phraseBoundaryWriter(textgrid_file, boundary_lab_file)