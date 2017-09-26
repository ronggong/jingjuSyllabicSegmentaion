# parameters of ODF onset detection function

# ODF method: 'jordi': Pons' CNN, 'jan': Schluter' CNN, 'jan_chan3'
mth_ODF         = 'jan'

# layer2 node number: 20 or 32
layer2          = 32

# late fusion: Bool
fusion          = False

# filter shape: 'temporal' or 'timbral' filter shape in Pons' CNN
filter_shape    = 'timbral'


# other params
fs = 44100
framesize_t = 0.025     # in second
hopsize_t   = 0.010

framesize   = int(round(framesize_t*fs))
hopsize     = int(round(hopsize_t*fs))

# MFCC params
highFrequencyBound = fs/2 if fs/2<11000 else 11000

varin                = {}
varin['N_feature']   = 40
varin['N_pattern']   = 21                # adjust this param, l in paper

# mfccBands feature half context window length
varin['nlen']        = 10

# parameters of viterbi
varin['delta_mode'] = 'proportion'
varin['delta']      = 0.35

varin['plot'] = True

varin['decoding'] = 'viterbi'