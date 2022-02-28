import nltk
import urllib.request
import zipfile
import re
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from pymongo import MongoClient
from gensim.models import Word2Vec, KeyedVectors

# nltk.download('punkt')

# 데이터 전처리
# 학습 데이터는 COMTRIS_AI 학습 데이터를 이용한다.


class Mongo():
    '''MongoDB Database Management'''

    def __init__(self):
        self.db_client = MongoClient(os.environ['COMTRIS_SERVER_MONGODB_URI'])
        self.db_cursor = self.db_client['COMTRIS']

    def client(self):
        '''DB client cursor 반환'''
        return self.db_client
    
    def cursor(self):
        '''RAAS cursor 반환'''
        return self.db_cursor

    def __del__(self):
        self.db_client.close()


DB = Mongo()

# 학습 데이터 로드
db_result = list(DB.cursor()['gallery'].find({"pass": 1}))
result = [post['join'] for post in db_result]

'''
size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
window = 컨텍스트 윈도우 크기
min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
workers = 학습을 위한 프로세스 수
sg = 0은 CBOW, 1은 Skip-gram.
'''
model = Word2Vec(sentences=result, size=6, window=6, min_count=5, workers=4, sg=0)
model_result = model.wv.most_similar("AMD 5800X")
print(model_result)