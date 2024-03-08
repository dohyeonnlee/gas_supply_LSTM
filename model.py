from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, mean_absolute_error



class Model:

    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.model = None

    def build(self):

        # 모델 설정 및 훈련

        # 최종선택 seq7 - layer3 - unit40
        # lstm layer 3 / unit 40,40,40 / activation = tanh
        # optimizer = adam / loss = mse / metrics = mae
        # 성능 개선 없을 경우를 위해 조기 멈춤(early stopping) 사용
        # val_loss, val_mae 사용가능
        
        early= EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model = Sequential()
        model.add(LSTM(units=40, activation='tanh', return_sequences=True, input_shape=(7,5)))
        model.add(LSTM(units=40, activation='tanh', return_sequences=True))
        model.add(LSTM(units=40, activation='tanh'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        model.fit(self.train_x, self.train_y, epochs=100, validation_split=0.2, callbacks=[early])

        self.model = model

    def evaluate(self):
        test_y = self.test_y
        pred_y = self.model.predict(self.test_x)
        
        print("test MSE", round(mean_squared_error(test_y, pred_y), 4))
        print("test MAE", round(mean_absolute_error(test_y, pred_y), 4))
        
        return pred_y