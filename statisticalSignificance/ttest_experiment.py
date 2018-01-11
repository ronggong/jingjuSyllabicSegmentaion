from scipy.stats import ttest_ind
import pickle
import os

# schluter
recall_precision_f1_a = pickle.load(open(os.path.join('./data/schluter/simpleWeighting', 'schluter_jan.pkl'), 'r'))
recall_precision_f1_b = pickle.load(open(os.path.join('./data/schluter/simpleWeighting', 'schluter_jordi_temporal_schluter.pkl'), 'r'))

f1_schluter_jan = [rpf[2] for rpf in recall_precision_f1_a]
f1_schluter_temporal = [rpf[2] for rpf in recall_precision_f1_b]

# # ismir
f1_ismir_jan_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_old+new_ismir_madmom_early_stopping_peakPickingMadmom.pkl'), 'r'))
f1_ismir_jan_v_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_old+new_ismir_madmom_early_stopping_viterbi_nolabel.pkl'), 'r'))
f1_ismir_jan_v_l = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jan_old+new_ismir_madmom_early_stopping_viterbi_label.pkl'), 'r'))

f1_ismir_temporal_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jordi_temporal_ismir_madmom_early_stopping_jan_params_peakPickingMadmom.pkl'), 'r'))
f1_ismir_temporal_v_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jordi_temporal_ismir_madmom_early_stopping_jan_params_viterbi_nolabel.pkl'), 'r'))
f1_ismir_temporal_v_l = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jordi_temporal_ismir_madmom_early_stopping_jan_params_viterbi_label.pkl'), 'r'))

# artist album filter
# f1_a = pickle.load(open(os.path.join('./data/', 'jan_old+new_artist_filter_madmom_early_stopping_peakPickingMadmom.pkl'), 'r'))
# f1_a = pickle.load(open(os.path.join('./data/', 'jan_old+new_artist_filter_madmom_early_stopping_viterbi_nolabel.pkl'), 'r'))
# f1_a = pickle.load(open(os.path.join('./data/', 'jordi_temporal_artist_filter_madmom_early_stopping_jan_params_peakPickingMadmom.pkl'), 'r'))
# f1_b = pickle.load(open(os.path.join('./data/', 'jordi_temporal_artist_filter_madmom_early_stopping_jan_params_viterbi_nolabel.pkl'), 'r'))

# viterbi label ismir

# viterbi label artist filter
# f1_a = pickle.load(open(os.path.join('./data/', 'jan_old+new_artist_filter_madmom_early_stopping_viterbi.pkl'), 'r'))
# f1_b = pickle.load(open(os.path.join('./data/', 'jordi_temporal_artist_filter_madmom_early_stopping_more_params_viterbi.pkl'), 'r'))

_, p = ttest_ind(f1_schluter_jan, f1_schluter_temporal, equal_var=False)

print('schluter jan temporal')
print(p)

def pValueAll(f1_jan_pp, f1_temporal_pp,
              f1_jan_v_nl, f1_temporal_v_nl,
              f1_jan_v_l, f1_temporal_v_l):

    _, p_jan_pp_temporal_pp = ttest_ind(f1_jan_pp, f1_temporal_pp, equal_var=False)

    _, p_jan_pp_jan_v_nl = ttest_ind(f1_jan_pp, f1_jan_v_nl, equal_var=False)

    _, p_jan_v_nl_temporal_v_nl = ttest_ind(f1_jan_v_nl, f1_temporal_v_nl, equal_var=False)

    _, p_temporal_pp_temporal_v_nl = ttest_ind(f1_temporal_pp, f1_temporal_v_nl, equal_var=False)

    _, p_jan_v_l_temporal_v_l = ttest_ind(f1_jan_v_l, f1_temporal_v_l, equal_var=False)

    return p_jan_pp_temporal_pp, \
           p_jan_pp_jan_v_nl, \
           p_jan_v_nl_temporal_v_nl, \
           p_temporal_pp_temporal_v_nl, \
           p_jan_v_l_temporal_v_l


def writePvalue2Txt(filename,
                    p_jan_pp_temporal_pp,
                    p_jan_pp_jan_v_nl,
                    p_jan_v_nl_temporal_v_nl,
                    p_temporal_pp_temporal_v_nl,
                    p_jan_v_l_temporal_v_l):
    with open(filename, 'w') as f:
        f.write('jan pp vs temporal pp')
        f.write('\n')
        f.write(str(p_jan_pp_temporal_pp))
        f.write('\n')
        f.write('jan pp vs jan viterbi nl')
        f.write('\n')
        f.write(str(p_jan_pp_jan_v_nl))
        f.write('\n')
        f.write('jan viterbi nl vs temporal viterbi nl')
        f.write('\n')
        f.write(str(p_jan_v_nl_temporal_v_nl))
        f.write('\n')
        f.write('temporal pp vs temporal viterbi nl')
        f.write('\n')
        f.write(str(p_temporal_pp_temporal_v_nl))
        f.write('\n')
        f.write('jan viterbi l vs temporal viterbi l')
        f.write('\n')
        f.write(str(p_jan_v_l_temporal_v_l))


p_jan_pp_temporal_pp, \
p_jan_pp_jan_v_nl, \
p_jan_v_nl_temporal_v_nl, \
p_temporal_pp_temporal_v_nl, \
p_jan_v_l_temporal_v_l = pValueAll(f1_ismir_jan_pp, f1_ismir_temporal_pp,
                                  f1_ismir_jan_v_nl, f1_ismir_temporal_v_nl,
                                  f1_ismir_jan_v_l, f1_ismir_temporal_v_l)


writePvalue2Txt('ismir.txt', p_jan_pp_temporal_pp,
                        p_jan_pp_jan_v_nl,
                        p_jan_v_nl_temporal_v_nl,
                        p_temporal_pp_temporal_v_nl,
                        p_jan_v_l_temporal_v_l)
