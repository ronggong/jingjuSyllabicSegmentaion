fs = 44100

hopsize_t = 0.010

varin = {}

varin['plot'] = False

varin['sample_weighting'] = 'simpleWeighting'

phrase_eval = False

filter_shape_0 = 'jan'

nfolds = 8

overlap = False

bidi = False

relu = False

deep = True

no_dense = True

overlap_str = '_overlap' if overlap else ''
phrase_str = '_phrase' if phrase_eval else ''
bidi_str = '_bidi_100' if bidi else ''
relu_str = '_relu' if relu else ''
deep_str = '_less_deep' if deep else ''
no_dense_str = '_jingju_no_dense' if no_dense else ''

weighting_str = 'simpleSampleWeighting' if varin['sample_weighting'] == 'simpleWeighting' else 'positiveThreeSampleWeighting'