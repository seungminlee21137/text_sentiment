import sys
import rhinoMorph

# 
def main(vect, grid, getData):
    print(f"main init::{getData}")

    rn = rhinoMorph.startRhino()
    new_input = getData

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
    resultMessage = "2";
    if result[0] == 0:
        print(f"{new_input}:{result[0]}: 부정")
        resultMessage = f"{new_input}:{result[0]}: 부정"
    if result[0] == '0':
        print(f"{new_input}::{result[0]}: 부정")
        resultMessage = f"{new_input}::{result[0]}: 부정"
    else:
        print(f"{new_input}:::{result[0]}: 긍정")
        resultMessage = f"{new_input}:::{result[0]}: 긍정"

    return resultMessage

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