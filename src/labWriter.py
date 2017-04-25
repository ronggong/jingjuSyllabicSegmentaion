import textgridParser

def boundaryLabWriter(boundaryList, outputFilename, label=False):
    '''
    Write the boundary list into .lab
    :param boundaryList:
    :param outputFilename:
    :return:
    '''

    with open(outputFilename, "wb") as lab_file:
        for list in boundaryList:
            if label:
                lab_file.write("{0:.4f} {1:.4f} {2}\n".format(list[0],list[1],list[2]))
            else:
                lab_file.write("{0:.4f} {1:.4f}\n".format(list[0],list[1]))

def phraseBoundaryWriter(textgrid_file, outputFilename):
    '''
    Write phrase boundary from textgrid into outputFilename, example: .syll.lab
    :param textgrid_file:
    :param outputFilename:
    :return:
    '''

    # read phrase list and utterance list
    lineList                    = textgridParser.textGrid2WordList(textgrid_file, whichTier='line')
    utteranceList               = textgridParser.textGrid2WordList(textgrid_file, whichTier='utterance')

    # parse lines of groundtruth
    nestedUtteranceLists, numLines, numUtterances = textgridParser.wordListsParseByLines(lineList, utteranceList)

    # phrase start, end time
    nonEmptyLineList            = []

    for list in nestedUtteranceLists:
        nonEmptyLineList.append(list[0])

    boundaryLabWriter(nonEmptyLineList, outputFilename)