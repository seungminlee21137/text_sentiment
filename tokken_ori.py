import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import os


# 다른 데이터셋 로드

import json
import pandas as pd


# 참조2:https://m.blog.naver.com/j7youngh/222966330839

# 온라인리드
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
# train_data = pd.read_table('ratings_train.txt')
# print('훈련용 리뷰 개수 :',len(train_data)) # 훈련용 리뷰 개수 출력

# {
# 	"word": "못미덥다",
# 	"word_root": "못",
# 	"polarity": "-1"
# },
with open('D:\\workspace/python_project/KoBERT_music_recomendation/data/SentiWord_info.json', encoding='utf-8-sig', mode='r') as f: 
  SentiWord_info = json.load(f)

sentiword_dic = pd.DataFrame(SentiWord_info)

print (len(sentiword_dic))

df = pd.DataFrame(columns=("review", "sentiment"))

data_text = []
data_senti = []
for i in range(10010, 10022):            
#for i in range(0, len(sentiword_dic)):            # 감성사전의 모든 단어를 하나씩 선택
    # if sentiword_dic.word[i] in token:              # 리뷰 문장에 감성 단어가 있는지 확인
        # sentiment += int(sentiword_dic.polarity[i])   # 감성단어가 있다면 극성값 합계를 구함.
    # df.loc[idx] = [token, sentiment]
    # print (sentiword_dic.word[i])
    # print (sentiword_dic.word_root[i])
    # print (sentiword_dic.polarity[i])

    data_text.append(sentiword_dic.word[i])
    senti_point = ""
    if (sentiword_dic.polarity[i] == "-1"):
        senti_point = "0"
    elif (sentiword_dic.polarity[i] == "-2"):
        senti_point = "0"
    elif (sentiword_dic.polarity[i] == "0"):
        senti_point = "1"
    elif (sentiword_dic.polarity[i] == "1"):
        senti_point = "1"
    elif (sentiword_dic.polarity[i] == "2"):
        senti_point = "1"
    
    data_senti.append(senti_point)
    

# def read_data(filename):
#     with open(filename, 'r', encoding="cp949") as f:
#         data = [line.split('\t') for line in f.read().splitlines()]
#         data = data[1:]
#     return data
print (data_text)
print (data_senti)

def read_data(filename):
    with open(filename, 'r', encoding="cp949") as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

data = read_data(r'D:\\workspace/python_project/KoBERT_music_recomendation/data/ratings_morphed.txt')
data_text = [line[1] for line in data]    
data_senti = [line[2] for line in data]

print (data_text[:20])
print (data_senti[:20])
print (len(data_text))

# # pip install sklearn.model
# from sklearn.model_selection import train_test_split
# train_data_text, test_data_text, train_data_senti, test_data_senti = train_test_split(
#         data_text,
#         data_senti,
#         stratify=data_senti,
#         test_size=0.3,
#         random_state=156
# )

# # pip install counter
# # Test와 Train이 잘 나누어졌는지 확인합니다.

# vect = TfidfVectorizer(min_df=5, ngram_range=(1,2)).fit(train_data_text)
# X_train = vect.transform(train_data_text)

# y_train = pd.Series(train_data_senti)
# scores = cross_val_score(LogisticRegression(solver="liblinear"), X_train, y_train, cv=5)
# #print('교차 검증 점수:', zscores)
# #print('교차 검증 점수 평균:', scores.mean())

# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.01, 0.1, 1, 3, 5]}
# grid = GridSearchCV(LogisticRegression(solver="liblinear"), param_grid, cv=5)
# grid.fit(X_train, y_train)
# #print("최고 교차 검증 점수:", round(grid.best_score_, 3))
# #print("최적의 매개변수:", grid.best_params_)
#=====================================================================================================

# # 6. 신규 데이터를 넣어주십시오.
# import rhinoMorph
# rn = rhinoMorph.startRhino()

# # print('rn\n',rn)

# new_input = '토게에 트리거하브옵 단일이랑 통일시키라고 피싸기 ㄱㄱ?'

# # 입력 데이터 형태소 분석하기
# inputdata = []
# morphed_input = rhinoMorph.onlyMorph_list(rn, new_input, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'])
# morphed_input = ' '.join(morphed_input)     # 한 개의 문자열로 만들기
# inputdata.append(morphed_input)             # 분석 결과를 리스트로 만들기
# X_input = vect.transform(inputdata)

# # // 0.0
# # print(float(grid.predict(X_input)))

# result = grid.predict(X_input)  # 0은 부정, 1은 긍정

# print(result)
# print(result[0])
# # print(type(result))

# if result[0] == 0:
#     print(f"{new_input}:{result[0]}: 부정")
# if result[0] == '0':
#     print(f"{new_input}::{result[0]}: 부정")
# else:
#     print(f"{new_input}:::{result[0]}: 긍정")

# # https://blog.naver.com/lingua/221537630069 파이썬 형태소분석기 RHINO
#=====================================================================================================