import os
import sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from flask import Flask
from flask import request

##################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# pip install sklearn.model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
#
from collections import Counter
#
import rhinoMorph
# 다른 데이터셋 로드
import json

def read_data(filename):
    with open(filename, 'r', encoding="cp949") as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

data = read_data(r'D:\\workspace/python_project/KoBERT_music_recomendation/data/ratings_morphed.txt')
data_text  = [line[1] for line in data]    
data_senti = [line[2] for line in data]



# with open('D:\\workspace/python_project/KoBERT_music_recomendation/data/SentiWord_info.json', encoding='utf-8-sig', mode='r') as f: 
#   SentiWord_info = json.load(f)
# sentiword_dic = pd.DataFrame(SentiWord_info)
#df = pd.DataFrame(columns=("review", "sentiment"))
# data_text = []
# data_senti = []
# 5개의 세분화 데이터를 2개로 병합
# for i in range(0, len(sentiword_dic)):
#     data_text.append(f"{sentiword_dic.word_root[i]} {sentiword_dic.word[i]}")

#     senti_point = ""
#     if (sentiword_dic.polarity[i] == "-1"):
#         senti_point = "0"
#     elif (sentiword_dic.polarity[i] == "-2"):
#         senti_point = "0"
#     elif (sentiword_dic.polarity[i] == "0"):
#         senti_point = "1"
#     elif (sentiword_dic.polarity[i] == "1"):
#         senti_point = "1"
#     elif (sentiword_dic.polarity[i] == "2"):
#         senti_point = "1"
    
#     data_senti.append(senti_point)

#     if(i > 4000 and i < 4020):
#         print(f"{i}_{sentiword_dic.word_root[i]}::{sentiword_dic.word[i]}_{sentiword_dic.polarity[i]}_{senti_point}_{senti_point == "1"}")
###############################################
# 학습 데이터셋 END
###############################################

train_data_text, test_data_text, train_data_senti, test_data_senti = train_test_split(
    data_text,
    data_senti,
    stratify=data_senti,
    test_size=0.3,
    random_state=156
)

# https://wikidocs.net/44249
# pip install counter
# Test와 Train이 잘 나누어졌는지 확인합니다.

vect = TfidfVectorizer(min_df=5, ngram_range=(1,2)).fit(train_data_text)
X_train = vect.transform(train_data_text)

y_train = pd.Series(train_data_senti)
# Logistic Regression (로지스틱 회귀)
scores = cross_val_score(LogisticRegression(solver="liblinear"), X_train, y_train, cv=5)
param_grid = {'C': [0.01, 0.1, 1, 3, 5]}
grid = GridSearchCV(LogisticRegression(solver="liblinear"), param_grid, cv=5)
grid.fit(X_train, y_train)

#
rn = rhinoMorph.startRhino()

server = Flask(__name__)

@server.route('/')
def home():
   return 'This is Home!'

@server.route('/tokken')
def tokken():
    # for arg in sys.argv:
    print(request.values.get('text'))

    params = request.values.get('text')

    #xx = default.main(vect, grid, request.values.get('text'))
    
    new_input = params

    # 입력 데이터 형태소 분석하기
    inputdata = []
    morphed_input = rhinoMorph.onlyMorph_list(rn, new_input, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'])
    morphed_input = ' '.join(morphed_input)     # 한 개의 문자열로 만들기
    inputdata.append(morphed_input)             # 분석 결과를 리스트로 만들기
    X_input = vect.transform(inputdata)

    print(X_input)

    result = grid.predict(X_input)  # 0은 부정, 1은 긍정
    
    print('교차 검증 점수:', scores)
    print('교차 검증 점수 평균:', scores.mean())
    print("최고 교차 검증 점수:", round(grid.best_score_, 3))
    print("최적의 매개변수:", grid.best_params_)

    print(result)
    
    # print(type(result))
    resultMessage = ""
    resultCode = 2
    
    if result[0] == 0:
        print(f"{new_input}:{result[0]}: 부정")
        resultMessage = "부정"
        resultCode = result[0]

    elif result[0] == '0':
        print(f"{new_input}::{result[0]}: 부정")
        resultMessage = "부정2"
        resultCode = result[0]

    else:
        print(f"{new_input}:::{result[0]}: 긍정")
        resultMessage = "긍정"
        resultCode = result[0]
   
    # return f'This [{resultMessage}]'
    return { 'text': params, 'resultCode': f"{resultCode}", 'resultMessage': resultMessage }

if __name__ == '__main__':  
    server.run('0.0.0.0',port=5000,debug=True)