import numpy as np

def peakPicking(obs):
    """
    Peak picking function written in paper
    Drum transcription via joint beat and drum modeling using convolutional recurrent neural networks
    :param obs:
    :return:
    """
    a = 2
    m = 2
    w = 2
    delta = 0.15
    len_conc = max(a,m)
    obs_extended = np.concatenate((np.array([0]*len_conc), obs))
    indices_peak = []
    for ii in range(len_conc, len(obs_extended)):
        if obs_extended[ii] == max(obs_extended[ii-m:ii+1]) \
            and obs_extended[ii] > np.mean(obs_extended[ii-a:ii+1]) + delta:

            if len(indices_peak)==0:
                indices_peak.append(ii-len_conc)
            else:
                if ii-len_conc-indices_peak[-1] > w:
                    indices_peak.append(ii-len_conc)
    indices_peak.append(len(obs)-1)
    return indices_peak