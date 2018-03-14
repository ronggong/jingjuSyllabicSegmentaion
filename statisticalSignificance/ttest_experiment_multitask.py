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
f1_artist_filter_less_deep_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_syllable_subset_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_joint_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_subset_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_joint_lw_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_subset_loss_weighted_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_joint_sv_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_subset_syllable_val_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_joint_svlw_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_subset_syllable_val_loss_weighted_peakPickingMadmom.pkl'), 'r'))

# # artist_filter viterbi no label
f1_artist_filter_less_deep_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_syllable_subset_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_joint_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_subset_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_joint_lw_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_subset_loss_weighted_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_joint_sv_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_subset_syllable_val_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_joint_svlw_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_artist_filter_joint_subset_syllable_val_loss_weighted_viterbi_nolabel.pkl'), 'r'))


def pValueAll(f1_jan, f1_joint, f1_joint_lw, f1_joint_sv, f1_joint_svlw):

    _, p_jan_joint = ttest_ind(f1_jan, f1_joint, equal_var=False)

    _, p_jan_joint_lw = ttest_ind(f1_jan, f1_joint_lw, equal_var=False)

    _, p_jan_joint_sv = ttest_ind(f1_jan, f1_joint_sv, equal_var=False)

    _, p_jan_joint_svlw = ttest_ind(f1_jan, f1_joint_svlw, equal_var=False)


    print(p_jan_joint)
    print(p_jan_joint_lw)
    print(p_jan_joint_sv)
    print(p_jan_joint_svlw)

    return p_jan_joint, p_jan_joint_lw, p_jan_joint_sv, p_jan_joint_svlw



def stat_artist_pp():
    pValueAll(f1_artist_filter_less_deep_pp,
              f1_artist_filter_joint_pp,
              f1_artist_filter_joint_lw_pp,
              f1_artist_filter_joint_sv_pp,
              f1_artist_filter_joint_svlw_pp)


def stat_artist_nl():
    pValueAll(f1_artist_filter_less_deep_nl,
              f1_artist_filter_joint_nl,
              f1_artist_filter_joint_lw_nl,
              f1_artist_filter_joint_sv_nl,
              f1_artist_filter_joint_svlw_nl)

if __name__ == '__main__':
    stat_artist_nl()