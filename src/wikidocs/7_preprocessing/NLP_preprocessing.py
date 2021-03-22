en_text = "Hello, My name is 837477 and developer."
kor_text = "안녕, 내 이름은 837477이고 개발자야."

# spaCy
# import spacy
# spacy_en = spacy.load('en')

# def tokenize(en_text):
#     return [tok.text for tok in spacy_en.tokenizer(en_text)]
# print(tokenize(en_text))


# nltk
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
# print(word_tokenize(en_text))


# mecab (형태소 토큰화)
from konlpy.tag import Mecab
tokenizer = Mecab()
print(tokenizer.morphs(kor_text))


# 단어 집합(Vocabulary) 생성
import urllib.request
import pandas as pd
from konlpy.tag import Mecab
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt') # 데이터프레임에 저장
data[:10]

print('전체 샘플의 수 : {}'.format(len(data)))

sample_data = data[:100] # 임의로 100개만 저장
sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
sample_data[:10]

# 불용어 정의
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

tokenizer = Mecab()
tokenized=[]
for sentence in sample_data['document']:
    temp = []
    temp = tokenizer.morphs(sentence) # 토큰화
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    tokenized.append(temp)
print(tokenized[:10])

vocab = FreqDist(np.hstack(tokenized))
print('단어 집합의 크기 : {}'.format(len(vocab)))

vocab_size = 500
# 상위 vocab_size개의 단어만 보존
vocab = vocab.most_common(vocab_size)
print('단어 집합의 크기 : {}'.format(len(vocab)))


# 각 단어에 고유한 정수 부여
word_to_index = {word[0] : index + 2 for index, word in enumerate(vocab)}
word_to_index['pad'] = 1
word_to_index['unk'] = 0

encoded = []
for line in tokenized: #입력 데이터에서 1줄씩 문장을 읽음
    temp = []
    for w in line: #각 줄에서 1개씩 글자를 읽음
      try:
        temp.append(word_to_index[w]) # 글자를 해당되는 정수로 변환
      except KeyError: # 단어 집합에 없는 단어일 경우 unk로 대체된다.
        temp.append(word_to_index['unk']) # unk의 인덱스로 변환

    encoded.append(temp)

print(encoded[:10])

