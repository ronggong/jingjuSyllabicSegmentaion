import vamp
import soundfile as sf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

def run_nnls_extraction(data, rate):
    """
    calculate chroma feature by NNLS vamp plug-in
    :param filename_wav:
    :return: chromadata, stepsize (sec.)
    """
    # print(len(data), rate)
    chroma = vamp.collect(data, rate, "nnls-chroma:nnls-chroma",output='chroma',step_size=1024, block_size=2048)
    stepsize, chromadata = chroma["matrix"]
    return chromadata, stepsize

def buildChroma(dict_score_line, sample_number_total):
    dict_chroma_dimension = {'A':0, 'A#':1, 'B-':1, 'B': 2, 'C':3, 'C#':4, 'D-':4, 'D':5, 'D#':6, 'E-':6,
                             'E':7, 'F':8, 'F#':9, 'G-':9, 'G':10, 'G#':11, 'A-':11}
    idx_syllable_onset = []
    chroma = np.empty((0, 12))
    length_total = sum([note['quarterLength'] for note in dict_score_line['notesWithRealRests']])
    for ii, note in enumerate(dict_score_line['notesWithRealRests']):
        sample_note = int(round((note['quarterLength']/float(length_total))*sample_number_total))

        if sample_note != 0:
            # if note contains a lyric, add the onset index
            if note['lyric']:
                idx_syllable_onset.append(chroma.shape[0])

            sample_chroma = np.zeros((sample_note, 12))
            if note['noteName']:
                if len(note['noteName']) > 2:
                    noteName = note['noteName'][:2]
                else:
                    noteName = note['noteName']
                sample_chroma[:, dict_chroma_dimension[noteName]] = 1.0
            chroma = np.concatenate((chroma, sample_chroma), axis=0)
    return chroma, idx_syllable_onset

def buildChroma_manualCorrectedCsv(syllables, notes, note_durations, sample_number_total):
    dict_chroma_dimension = {'A':0, 'A#':1, 'B-':1, 'B': 2, 'C':3, 'C#':4, 'D-':4, 'D':5, 'D#':6, 'E-':6,
                             'E':7, 'F':8, 'F#':9, 'G-':9, 'G':10, 'G#':11, 'A-':11}
    idx_syllable_onset = []
    chroma = np.empty((0, 12))
    length_total = sum([float(nd) for nd in note_durations if len(nd)])
    for ii in range(len(syllables)):
        if len(note_durations[ii]):
            sample_note = int(round((float(note_durations[ii])/length_total)*sample_number_total))

            if sample_note != 0:
                # if note contains a lyric, add the onset index
                if syllables[ii]:
                    idx_syllable_onset.append(chroma.shape[0])

                sample_chroma = np.zeros((sample_note, 12))
                note_ii = notes[ii].split('_')[0]
                if note_ii:
                    if len(note_ii) > 2:
                        noteName = note_ii[:2]
                    else:
                        noteName = note_ii
                    sample_chroma[:, dict_chroma_dimension[noteName]] = 1.0
                chroma = np.concatenate((chroma, sample_chroma), axis=0)
    return chroma, idx_syllable_onset


def dist_point2line(point, k):
    """
    distance from a point to a line
    :param point: (x, y)
    :param k: the slope of the line
    :return:
    """
    dist = np.abs(k*point[0]+(-1)*point[1])/np.square(k**2+1)
    return dist

def chroma_aligned_feature(chroma_ref, chroma_stu):
    """
    calculate chroma aligned feature
    :param chroma_ref:
    :param chroma_stu:
    :param path:
    :return: first col: alignment path to diagonal distance, second col: chroma aligned distance
    """
    # dtw get alignment path
    _, path =fastdtw(chroma_ref, chroma_stu, dist=euclidean)
    # calculate the distance from the alignment path point to the diagonal
    dist_path_to_diag = [dist_point2line(point, chroma_ref.shape[0]/float(chroma_stu.shape[0])) for point in path]
    # the distance between two chroma vectors aligned
    dist_chroma_aligned = [euclidean(chroma_ref[point[0], :], chroma_stu[point[1], :]) for point in path]
    # stack the alignment path distance and the chroma distance
    feature_chroma_aligned = np.vstack((dist_path_to_diag, dist_chroma_aligned))

    return feature_chroma_aligned.transpose()

if __name__ == '__main__':
    filename_test_wav_ref = '/home/gong/Documents/MTG/dataset/MAST2grade/51_mel1/reference/51_mel1_ref102585.wav'
    filename_test_wav_stu = '/home/gong/Documents/MTG/dataset/MAST2grade/51_mel1/performances/51_mel1_per101559_fail.wav'

    chroma_ref, step_size_ref = run_nnls_extraction(filename_test_wav_ref)
    chroma_stu, step_size_stu = run_nnls_extraction(filename_test_wav_stu)

    print('reference chroma shape:', chroma_ref.shape)
    print('student chroma shape:', chroma_stu.shape)

    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # plt.subplot(2, 1, 1)
    # plt.imshow(np.transpose(chroma_ref))
    # plt.title('reference up, student bottom')
    # plt.subplot(2, 1, 2)
    # plt.imshow(np.transpose(chroma_stu))
    # plt.show()