import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

### 데이터셋 불러오기
dataset = pd.read_csv('./data/chap02/data/car_evaluation.csv')
# print(dataset.head())  # 데이터 프레임 내의 첫 n줄을 출력해서 데이터의 내용을 확인

### 데이터 전처리
categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']   # 예제 데이터셋의 column들의 목록
for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')    # astype() 메서드를 이용하여 데이터를 범주형으로 변환

# 범주형 데이터를 텐서로 변환하기 위해 다음과 같은 절차가 필요함
# 범주형 데이터 -> dataset[category] -> 넘파이 배열(NumPy array) -> 텐서(Tensor)
# 범주형 데이터(단어)를 숫자(넘파이 배열)로 변환하기 위해 cat.codes를 사용
price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

# np.stack은 두 개 이상의 넘파이 객체를 합칠 때 사용
categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)
# print(categorical_data[:10])

# 이제 torch 모듈을 사용해 배열을 텐서로 변환
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
# print(categorical_data[:10])

# 마지막으로 레이블(outputs)로 사용할 column에 대해서도 텐서로 변환
# get_dummies는 가변수(dummy variable)로 만들어주는 함수 -> 문자를 (0,1)로 바꿔주는 것
outputs = pd.get_dummies(dataset.output)
outputs = outputs.values
outputs = torch.tensor(outputs).flatten()   # 1차원 텐서로 변환
# 텐서의 차원을 1차원으로 바꾸는 메서드(ravel(), reshape(-1), flatten())

# print(categorical_data.shape)
# print(outputs.shape)

# 배열을 N차원으로 변환하기 위해 모든 범주형 column에 대해 임베딩 크기를 정의
categorical_column_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]
# print(categorical_embedding_sizes)

# 데이터셋을 훈련과 테스트 용도로 분리
total_records = 1728
test_records = int(total_records * 0.2) # 전체 데이터 중 20%를 테스트 용도로 사용

categorical_train_data = categorical_data[:total_records-test_records]
categorical_test_data = categorical_data[total_records-test_records:total_records]
train_outputs = outputs[:total_records-test_records]
test_outputs = outputs[total_records-test_records:total_records]

