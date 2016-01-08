# -*- coding: utf-8 -*-
import sys, os

currentPath = os.path.dirname(__file__)
utilsPath = os.path.join(currentPath, 'utils')
sys.path.append(utilsPath)

import textgrid as tgp

def TextGrid2WordList(textgrid_file, whichTier = 'pinyin'):
    '''
    parse textGrid into a python list of tokens 
    @param whichTier : 'pinyin' default tier name  
    '''	
    if not os.path.isfile(textgrid_file): raise Exception("file {} not found".format(textgrid_file))
    beginTsAndWordList=[]
	
    par_obj = tgp.TextGrid.loadUTF16(textgrid_file)	#loading the object	
    tiers= tgp.TextGrid._find_tiers(par_obj)	#finding existing tiers		
	
    isTierFound = False
    for tier in tiers:
        tierName= tier.tier_name().replace('.','')
        #iterating over tiers and selecting the one specified
        if tierName == whichTier:
			isTierFound = True
			#this function parse the file nicely and return cool tuples
			tier_details = tier.make_simple_transcript();

			for line in tier_details:
				beginTsAndWordList.append([float(line[0]), float(line[1]), line[2]])
    if not isTierFound:
		raise Exception('tier in file {0} might not be named correctly. Name it {1}' .format(textgrid_file,  WordLevelEvaluator.tier_names[whichTier]))
    return beginTsAndWordList

def Line2WordList(line, entireWordList):
	'''
	
	'''