from src.textgridParser import textGrid2WordList, wordListsParseByLines
import matplotlib.pyplot as plt
from scipy.misc import factorial
from scipy.optimize import curve_fit
from scipy.stats import gamma,expon
import numpy as np
from src.parameters import *
from src.file_path_jingju import *
from src.phonemeMap import dic_pho_map
import json
import os


def phoDurCollection(full_path_textgrids):
    '''
    collect durations of pho into dictionary
    :param recordings:
    :return:
    '''
    dict_duration_pho = {}
    for full_path_textgrid in full_path_textgrids:

        lineList = textGrid2WordList(full_path_textgrid, whichTier='dian')
        utteranceList = textGrid2WordList(full_path_textgrid, whichTier='details')

        # parse lines of groundtruth
        nestedPhonemeLists, _, _ = wordListsParseByLines(lineList, utteranceList)


        for pho in nestedPhonemeLists:
            for p in pho[1]:
                dur_pho = p[1] - p[0]
                sampa_pho = dic_pho_map[p[2]]

                if sampa_pho not in dict_duration_pho.keys():
                    dict_duration_pho[sampa_pho] = [dur_pho]
                else:
                    dict_duration_pho[sampa_pho].append(dur_pho)
    return dict_duration_pho

def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

def durPhoDistribution(array_durPho,sampa_pho,plot=False):
    '''
    pho durations histogram
    :param array_durPho:
    :return:
    '''

    # plt.figure(figsize=(10, 6))

    # integer bin edges
    offset_bin = 0.005
    bins = np.arange(0, max(array_durPho)+2, 2*offset_bin) - offset_bin

    # histogram
    entries, bin_edges, patches = plt.hist(array_durPho, bins=bins, normed=True, fc=(0, 0, 1, 0.7),label='pho: '+sampa_pho+' dur histogram')

    # centroid duration
    bin_centres = bin_edges-offset_bin
    bin_centres = bin_centres[:-1]
    centroid = np.sum(bin_centres*entries)/np.sum(entries)

    ##-- fit with poisson distribution
    # bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
    #
    # parameters, cov_matrix = curve_fit(poisson, bin_middles, entries)
    #
    # x = np.linspace(0, max(array_durPho), 1000)
    # x = np.arange(0,max(array_durPho),hopsize_t)
    #
    # p = poisson(x, *parameters)

    ##-- fit with gamma distribution

    # discard some outlier durations by applying 2 standard deviations interval
    mean_array_durPho=np.mean(array_durPho)
    std_array_durPho=np.std(array_durPho)
    index_keep = np.where(array_durPho<mean_array_durPho+2*std_array_durPho)
    array_durPho_keep = array_durPho[index_keep]

    # # discard according to pho
    # if dataset == 'qmLonUpfLaosheng':
    #     if sampa_pho == 'in':
    #         array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<2.5)]
    #     elif sampa_pho == '@n':
    #         array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<3)]
    #     elif sampa_pho == 'eI^':
    #         array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<1.5)]
    #     elif sampa_pho == 'EnEn':
    #         array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<2.0)]
    #     elif sampa_pho == 'UN':
    #         array_durPho_keep = array_durPho_keep[np.where(array_durPho_keep<2.5)]

    # step is the hopsize_t, corresponding to each frame
    # maximum length is the 4 times of the effective length
    x = np.arange(0, 8*max(array_durPho_keep),hopsize_t)

    param   = gamma.fit(array_durPho_keep,floc = 0)
    y       = gamma.pdf(x, *param)
    # y = expon.pdf(x)

    if plot:
        # plt.plot(x,p,'r',linewidth=2,label='Poisson distribution fitting curve')

        # plt.plot(x, y, 'r-', lw=2, alpha=0.6, label='gamma pdf')
        plt.axvline(centroid, linewidth = 3, color = 'r')
        plt.legend(fontsize=18)
        plt.xlabel('Pho duration distribution ',fontsize=18)
        plt.ylabel('Probability',fontsize=18)
        plt.axis('tight')
        plt.tight_layout()
        plt.show()

    y /= np.sum(y)

    return y.tolist(),centroid

if __name__ == '__main__':

    full_path_textgrids = []

    for rec in queenMary_Recordings_train_fem:
        full_path_textgrids.append(os.path.join(textgrid_path_dan, dict_name_mapping_dan_qm[rec] + '.textgrid'))

    for rec in queenMary_Recordings_train_male:
        full_path_textgrids.append(
            os.path.join(textgrid_path_laosheng, dict_name_mapping_laosheng_qm[rec] + '.textgrid'))

    for rec in bcn_Recordings_train_male:
        full_path_textgrids.append(
            os.path.join(textgrid_path_laosheng, dict_name_mapping_laosheng_bcn[rec] + '.textgrid'))

    for rec in bcn_Recordings_train_fem:
        full_path_textgrids.append(os.path.join(textgrid_path_dan, dict_name_mapping_dan_bcn[rec] + '.textgrid'))

    for rec in london_Recordings_train_male:
        full_path_textgrids.append(
            os.path.join(textgrid_path_laosheng, dict_name_mapping_laosheng_london[rec] + '.textgrid'))

    for rec in london_Recordings_train_fem:
        full_path_textgrids.append(os.path.join(textgrid_path_dan, dict_name_mapping_dan_london[rec] + '.textgrid'))

    dict_duration_pho = phoDurCollection(full_path_textgrids)

    dict_centroid_dur = {}
    dict_dur_dist = {}
    for pho in dict_duration_pho:
        durDist,centroid_dur = durPhoDistribution(np.array(dict_duration_pho[pho]),pho,plot=True)
        dict_centroid_dur[pho]  = centroid_dur
        dict_dur_dist[pho]      = durDist # the first proba is always 0

    with open('dict_centroid_dur.json','wb') as outfile:
        json.dump(dict_centroid_dur,outfile)

    # with open('dict_dur_dist_'+dataset+'.json','wb') as outfile:
    #     json.dump(dict_dur_dist,outfile)

