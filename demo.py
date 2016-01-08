# -*- coding: utf-8 -*-
import sys, os

sys.path.append(os.path.realpath('./src/'))

import textgridParser

# ----------------------------------------------------------------------
# a capella dataset
# 1. London recordings /Users/gong/Documents/MTG document/Jingju arias/londonRecording
# 2. Queen Mary dataset /Users/gong/Documents/MTG document/Jingju arias/QueenMary/jingjuSingingMono
# 3. Upf recordings /Users/gong/Documents/MTG document/Jingju arias/bcnRecording
# 4. source separation /Users/gong/Documents/MTG document/Jingju arias/cleanSinging

# test the textgridParser
textgrid_file = './testFiles/textgrid/shiwenhui_tingxiongyan.TextGrid'
print textgridParser.TextGrid2WordList(textgrid_file, whichTier='line')






