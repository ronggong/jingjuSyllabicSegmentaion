import os

def lab2WordList(lab_file):
    '''
    Parse Lab file into python list
    :param lab_file: Lab file path
    :return:
    '''

    if not os.path.isfile(lab_file): raise Exception("file {} not found".format(lab_file))
    with open(lab_file) as f:
        lineList = [line.rstrip() for line in f]
        dataList = []
        for l in lineList:
            startTime, endTime, label = l.split()
            if label != '##':
                dataList.append([float(startTime), float(endTime), label])

    return dataList