from scipy.stats import ttest_ind
import pickle
import os

"""
Jan less deep
Jan subset joint
Jan subset joint loss weighted
Jan subset joint syllable val loss early stopping
Jan subset joint syllable val loss early stopping loss weighted
"""

# # artist_filter peak picking
f1_artist_filter_less_deep_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_less_deep_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_pretrained_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_less_deep_pretrained_schluter_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_deep_feature_extraction_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_less_deep_deep_feature_extraction_schluter_peakPickingMadmom.pkl'), 'r'))

# # artist_filter viterbi no label
f1_artist_filter_less_deep_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_less_deep_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_pretrained_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_less_deep_pretrained_schluter_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_deep_feature_extraction_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_less_deep_deep_feature_extraction_schluter_viterbi_nolabel.pkl'), 'r'))


def pValueAll(f1_jan, f1_pretrained, f1_deep_feature_extraction):

    _, p_jan_pretrained = ttest_ind(f1_jan, f1_pretrained, equal_var=False)

    _, p_jan_deep_feature_extraction = ttest_ind(f1_jan, f1_deep_feature_extraction, equal_var=False)

    print(p_jan_pretrained)
    print(p_jan_deep_feature_extraction)

    return p_jan_pretrained, p_jan_deep_feature_extraction



def stat_artist_pp():
    pValueAll(f1_artist_filter_less_deep_pp,
              f1_artist_filter_pretrained_pp,
              f1_artist_filter_deep_feature_extraction_pp)


def stat_artist_nl():
    pValueAll(f1_artist_filter_less_deep_nl,
              f1_artist_filter_pretrained_nl,
              f1_artist_filter_deep_feature_extraction_nl)

if __name__ == '__main__':
    stat_artist_nl()