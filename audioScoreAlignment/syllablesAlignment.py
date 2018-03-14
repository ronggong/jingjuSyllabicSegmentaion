import string
import numpy as np

def convertSyl2Letters(syllables0, syllables1):
    """
    convert syllable lists to letter string
    :param syllables0:
    :param syllables1:
    :return:
    """
    dict_letters2syl = {}
    dict_syl2letters = {}
    ascii_letters = string.ascii_letters
    for ii, syl in enumerate(list(set(syllables0+syllables1))):
        dict_letters2syl[ascii_letters[ii]] = syl
        dict_syl2letters[syl] = ascii_letters[ii]

    syllables0_converted = ''.join([dict_syl2letters[syl] for syl in syllables0])
    syllables1_converted = ''.join([dict_syl2letters[syl] for syl in syllables1])
    return syllables0_converted, syllables1_converted, dict_letters2syl

def adjustSyllableDurByAlignment(durations, syllables_score_aligned, syllables_ground_truth_aligned):
    """
    As function name
    :param durations:
    :param syllables_score_aligned:
    :param syllables_ground_truth_aligned:
    :return:
    """
    # first insert values according to syllables_score_aligned insertions
    minimum_dur = min(durations)
    for ii, char in enumerate(syllables_score_aligned):
        if char == '-':
            durations.insert(ii, minimum_dur)
    # remove values according to groundtruth insertions
    for ii, char in reversed(list(enumerate(syllables_ground_truth_aligned))):
        if char == '-':
            if ii != 0:
                durations[ii-1] += durations[ii]
            else:
                durations[ii+1] += durations[ii]
            durations.pop(ii)
    return durations


def removeUnvoicedAndGetIdx(pitchtrack):
    """

    :param pitchtrack:
    :return:
    """
    idx_removed2entire = {}
    pitchtrack_unvoiced_removed = []
    for ii, pt in enumerate(pitchtrack):
        if pt > 0:
            idx_removed2entire[str(len(pitchtrack_unvoiced_removed))] = ii
            pitchtrack_unvoiced_removed.append(pt)
    return np.array(pitchtrack_unvoiced_removed), idx_removed2entire


def plotAlignment(pitchtrack_score, pitchtrack_audio, alignement_path, idx_syllable_onset_score, idx_removed2entire, ratio_interp):
    import matplotlib.pyplot as plt
    from src.utilFunctions import pitchtrackInterp

    if len(pitchtrack_audio) != len(pitchtrack_score):
        ratio_interp_audio = len(pitchtrack_audio)/float(len(pitchtrack_score))
        pitchtrack_score = np.array(pitchtrackInterp(pitchtrack_score, len(pitchtrack_audio)))
    else:
        ratio_interp_audio = 1.0

    down_shift = 1000
    plt.figure()
    plt.plot(pitchtrack_score)
    plt.plot(pitchtrack_audio - down_shift)

    print(len(idx_syllable_onset_score))

    idx_syllable_onset_score_ignore = []
    for pair in alignement_path:
        if pair[1] in idx_syllable_onset_score and pair[1] not in idx_syllable_onset_score_ignore:
            # get the corresponding aligned frame,
            # compensate the shrink, map this value to the original pitch track with silence
            x = [int(pair[1]*ratio_interp_audio), idx_removed2entire[str(int(pair[0] / ratio_interp))]]
            y = [pitchtrack_score[x[0]], pitchtrack_audio[x[1]]-down_shift]
            idx_syllable_onset_score_ignore.append(pair[1])
            plt.plot(x, y)

    plt.show()


def plotAlignmentChroma(chroma_score, chroma_audio, alignement_path, distance, idx_syllable_onset_score):
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    from matplotlib import lines

    # print(chroma_score.shape)
    # print(chroma_audio.shape)
    # print(idx_syllable_onset_score)

    fig = plt.figure(figsize=(5, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    idx_syllable_onset_score_ignore = []
    for pair in alignement_path:
        if pair[1] in idx_syllable_onset_score and pair[1] not in idx_syllable_onset_score_ignore:
            # get the corresponding aligned frame,
            # compensate the shrink, map this value to the original pitch track with silence
            x = [pair[1], pair[0]]
            y = [11, 0]
            idx_syllable_onset_score_ignore.append(pair[1])

            # print([y[0], x[0]])
            # print([y[1], x[1]])
            xy1 = [x[0], y[0]]
            xy2 = [x[1], y[1]]
            con = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA="data", coordsB="data",
                                  axesA=ax2, axesB=ax1, color="red")
            ax2.add_artist(con)

            idx_syllable_onset_score_ignore.append(pair[1])

    ax1.imshow(chroma_score)
    ax2.imshow(chroma_audio)

    plt.title(str(distance))

    plt.show()


def plotAlignmentPitchtrack(pt_score, pt_audio, idx_syllable_onset_score, idx_syllable_onset_audio):
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    # print(chroma_score.shape)
    # print(chroma_audio.shape)
    # print(idx_syllable_onset_score)

    fig = plt.figure(figsize=(5, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    idx_syllable_onset_score_ignore = []
    for ii in range(len(idx_syllable_onset_score)):
        # get the corresponding aligned frame,
        # compensate the shrink, map this value to the original pitch track with silence
        x = [idx_syllable_onset_score[ii], idx_syllable_onset_audio[ii]]
        y = [pt_score[x[0]], pt_audio[x[1]]]

        # print([y[0], x[0]])
        # print([y[1], x[1]])
        xy1 = [x[0], y[0]]
        xy2 = [x[1], y[1]]
        con = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)

    ax1.plot(pt_score)
    ax2.plot(pt_audio)


    plt.show()