import labWriter
import os
from src.filePath import *

for idx, recording_name in enumerate(queenMary_Recordings):

    textgrid_file       = os.path.join(aCapella_root, queenMarydataset_path, annotation_path,   recording_name+'.TextGrid')
    boundary_lab_file   = os.path.join(aCapella_root, queenMarydataset_path, segPhrase_path,    recording_name+'.lab')

    labWriter.phraseBoundaryWriter(textgrid_file, boundary_lab_file)

for idx, recording_name in enumerate(london_Recordings):

    textgrid_file       = os.path.join(aCapella_root, londonRecording_path, annotation_path,   recording_name+'.TextGrid')
    boundary_lab_file   = os.path.join(aCapella_root, londonRecording_path, segPhrase_path,    recording_name+'.lab')

    labWriter.phraseBoundaryWriter(textgrid_file, boundary_lab_file)

for idx, recording_name in enumerate(bcn_Recordings):

    textgrid_file       = os.path.join(aCapella_root, bcnRecording_path, annotation_path,   recording_name+'.TextGrid')
    boundary_lab_file   = os.path.join(aCapella_root, bcnRecording_path, segPhrase_path,    recording_name+'.lab')

    labWriter.phraseBoundaryWriter(textgrid_file, boundary_lab_file)