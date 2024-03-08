import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class Preprocessing:

    def __init__(self, data):
        self.data = data
        self.dfx = None
        self.dfy = None
        self.scaler = None

    def data_preprocess(self):
        df = self.data

        # 구분 알파벳 -> 숫자로 라벨링
        # 'A', 'B', 'C', 'D', 'E', 'G', 'H' -> 0-6
        lbe = LabelEncoder()
        df['구분'] = lbe.fit_transform(df['구분'])


        # 연월일 dateime으로 형변환 후 연/월/일/요일 로 컬럼 나누기
        # weekday - 요일 반환(0-6: 월-일)
        
        df['연월일'] = pd.to_datetime(df['연월일'])
        df['year'] = df['연월일'].dt.year
        df['month'] = df['연월일'].dt.month
        df['day'] = df['연월일'].dt.day
        df['weekday'] = df['연월일'].dt.weekday


        # 범주형 데이터를 제외한 공급량 값을 minmaxscaler를 통해 전처리

        dfx = df[['구분', 'month', 'day', 'weekday', '시간', '공급량(톤)']]

        scaler = MinMaxScaler()
        dfx['공급량(톤)'] = scaler.fit_transform(dfx[['공급량(톤)']])

        self.dfx = dfx
        self.dfy = dfx[['공급량(톤)']] # 예측값

        # 다시 원래 값으로 되돌릴 때 사용
        self.scaler = scaler

        
    def input_preprocess(self):

        dfx = self.dfx[['구분', 'month', 'day', 'weekday', '시간']]

        # 모든 값들을 일렬로 list화, 
        # 이후 원하는 값(한번에 학습할 수, 아래 window_size)만큼 잘라서 x값으로 만듦

        x = dfx.values.tolist()
        y = self.dfy.values.tolist()


        # LSTM 모델에 맞도록 input 데이터 전처리
        # 일주일(7일)의 데이터로 다음날의 공급량을 예측할 수 있게 전처리

        window_size = 7
        data_x, data_y = [], []

        for i in range(len(y) - window_size):
            _x = x[i:i+window_size]
            _y = y[i+window_size]
            data_x.append(_x)
            data_y.append(_y)


        # 2018년 전까지의 데이터(2013-2017) 인덱스 306768
        # 인덱스로 구분하여 train, test split
            
        train_size = 306768
        train_x = np.array(data_x[0 : train_size])
        train_y = np.array(data_y[0 : train_size])

        # test_size = len(data_y) - train_size
        test_x = np.array(data_x[train_size : len(data_x)])
        test_y = np.array(data_y[train_size : len(data_y)])

        print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
        print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)

        return train_x, train_y, test_x, test_y
