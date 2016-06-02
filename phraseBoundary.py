import labWriter
import os

# aCapella root
aCapella_root   = '/Users/gong/Documents/MTG document/Jingju arias/aCapella/'

# dataset path
queenMarydataset_path    = 'QueenMary/jingjuSingingMono'
londonRecording_path     = 'londonRecording'
bcnRecording_path        = 'bcnRecording'

segPhrase_path      = 'segPhrase'
annotation_path     = 'annotation'

# queen mary recordings
queenMaryFem_01_Recordings = ['fem_01/neg_1', 'fem_01/neg_3', 'fem_01/neg_5', 'fem_01/pos_1', 'fem_01/pos_3', 'fem_01/pos_5', 'fem_01/pos_7']
queenMaryFem_10_Recordings = ['fem_10/pos_1', 'fem_10/pos_3']
queenMaryFem_11_Recordings = ['fem_11/pos_1']

queenMaryMale_01_Recordings = ['male_01/neg_1','male_01/neg_2','male_01/neg_3','male_01/neg_4','male_01/neg_5',
                            'male_01/pos_1','male_01/pos_2','male_01/pos_3','male_01/pos_4','male_01/pos_5','male_01/pos_6']
queenMaryMale_02_Recordings = ['male_02/neg_1']
queenMaryMale_12_Recordings = ['male_12/neg_1']
queenMaryMale_13_Recordings = ['male_13/pos_1', 'male_13/pos_3']

queenMary_Recordings        = queenMaryFem_01_Recordings + queenMaryFem_10_Recordings + queenMaryFem_11_Recordings + \
    queenMaryMale_01_Recordings + queenMaryMale_02_Recordings + queenMaryMale_12_Recordings + queenMaryMale_13_Recordings

# london recordings
londonDan_Recordings        = ['Dan-01', 'Dan-02', 'Dan-03', 'Dan-04']
londonLaosheng_Recordings   = ['Laosheng-01', 'Laosheng-02', 'Laosheng-03', 'Laosheng-04']

london_Recordings           = londonDan_Recordings + londonLaosheng_Recordings

# bcn recordings
bcnDan_Recordings           = ['001', '007']
bcnLaosheng_Recordings      = ['003', '004', '005', '008']

bcn_Recordings              = bcnDan_Recordings + bcnLaosheng_Recordings

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