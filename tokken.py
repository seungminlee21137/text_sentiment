import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# df = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/tripadviser_review.csv")
# df = pd.read_clipboard("D:\\workspace/python_project/KoBERT_music_recomendation/data/SentiWord_info")
# df.head()
# print(df)

# https://nicola-ml.tistory.com/63
# Python 머신러닝, 한글 감정분석을 위한 리뷰 분석 : 프로그램부터 실전적용까지 (rhinoMorph이용)
import os

def read_data(filename):
    with open(filename, 'r', encoding="cp949") as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

data = read_data(r'D:\\workspace/python_project/KoBERT_music_recomendation/data/ratings_morphed.txt')
data_text = [line[1] for line in data]    
data_senti = [line[2] for line in data]

# pip install sklearn.model
from sklearn.model_selection import train_test_split
train_data_text, test_data_text, train_data_senti, test_data_senti = train_test_split(
        data_text,
        data_senti,
        stratify=data_senti,
        test_size=0.3,
        random_state=156
)

# pip install counter
# Test와 Train이 잘 나누어졌는지 확인합니다.
from collections import Counter
# train_data_senti_freq = Counter(train_data_senti)
# print('train_data_senti_freq:', train_data_senti_freq)

# test_data_senti_freq = Counter(test_data_senti)
# print('test_data_senti_freq:', test_data_senti_freq)


# from sklearn.feature_extraction.text import CountVectorizer
# vect = CountVectorizer(min_df=5).fit(train_data_text)
# X_train = vect.transform(train_data_text)

# feature_names = vect.get_feature_names()
# print("특성 개수:", len(feature_names))
# print("처음 20개 특성:\m", feature_names[:20])
# 만약 Tfid 벡터화를 원한다면 하기 내용으로 진행하면 됩니다.
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=5, ngram_range=(1,2)).fit(train_data_text)
X_train = vect.transform(train_data_text)

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
y_train = pd.Series(train_data_senti)
scores = cross_val_score(LogisticRegression(solver="liblinear"), X_train, y_train, cv=5)
#print('교차 검증 점수:', scores)
#print('교차 검증 점수 평균:', scores.mean())

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 3, 5]}
grid = GridSearchCV(LogisticRegression(solver="liblinear"), param_grid, cv=5)
grid.fit(X_train, y_train)
#print("최고 교차 검증 점수:", round(grid.best_score_, 3))
#print("최적의 매개변수:", grid.best_params_)


# 6. 신규 데이터를 넣어주십시오.
import rhinoMorph
rn = rhinoMorph.startRhino()

# print('rn\n',rn)

new_input = '토게에 트리거하브옵 단일이랑 통일시키라고 피싸기 ㄱㄱ?'

# 입력 데이터 형태소 분석하기
inputdata = []
morphed_input = rhinoMorph.onlyMorph_list(rn, new_input, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'])
morphed_input = ' '.join(morphed_input)     # 한 개의 문자열로 만들기
inputdata.append(morphed_input)             # 분석 결과를 리스트로 만들기
X_input = vect.transform(inputdata)

# // 0.0
# print(float(grid.predict(X_input)))

result = grid.predict(X_input)  # 0은 부정, 1은 긍정

print(result)
print(result[0])
# print(type(result))

if result[0] == 0:
    print(f"{new_input}:{result[0]}: 부정")
if result[0] == '0':
    print(f"{new_input}::{result[0]}: 부정")
else:
    print(f"{new_input}:::{result[0]}: 긍정")

# https://blog.naver.com/lingua/221537630069 파이썬 형태소분석기 RHINO
