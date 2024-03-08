import pandas as pd

from preprocessing import Preprocessing
from model import Model
from visualization import visualization

def main():

    # 데이터 가져오기
    
    file_name = "한국가스공사_시간별 공급량_20181231.csv"
    
    # data = pd.read_csv(f".//data{file_name}", encoding='cp949')
    data = pd.read_csv(f"C:\python\DL\gas\data\{file_name}", encoding='cp949')
    
    print(data)  

    # 전처리
    
    dp = Preprocessing(data)
    dp.data_preprocess()
    train_x, train_y, test_x, test_y = dp.input_preprocess()


    # 모델 학습
    # 2013-2017 데이터 학습

    model = Model(train_x, train_y, test_x, test_y)
    model.build()

    # 모델 예측 및 평가
    # 2018 데이터로 공급량 예측, 예측값 평가

    pred_y = model.evaluate()


    # 실제 데이터와 예측 데이터 시각화
    # scale한 데이터를 다시 원래 값으로 변환

    test_y_original = dp.scaler.inverse_transform(test_y)
    pred_y_original = dp.scaler.inverse_transform(pred_y)

    visualization(test_y_original, pred_y_original)



if __name__ == "__main__":
    main()