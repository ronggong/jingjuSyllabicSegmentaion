from src.filePath import *
from src.scoreParser import writeCsvPinyin
from src.trainTestSeparation import getTestTrainRecordingsArtist
import os

def insert_pinyin_2_csv_nacta2017_dataset():
    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtist()
    for artist_path, rn in testNacta2017:

        if not os.path.exists(join(nacta2017_score_pinyin_path, artist_path)):
            os.makedirs(join(nacta2017_score_pinyin_path, artist_path))

        score_file  = join(nacta2017_score_path, artist_path, rn+'.csv')
        score_pinyin_file  = join(nacta2017_score_pinyin_path, artist_path, rn+'.csv')
        writeCsvPinyin(score_file,score_pinyin_file)

# Don't run again this, or you will erase all the scores with pinyin

# if __name__ == '__main__':
#     insert_pinyin_2_csv_nacta2017_dataset()