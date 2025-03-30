import math
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt
from src.stock_model.ai_models.base_model import AIModel
plt.style.use('fivethirtyeight')

class CnnLSTMHybrid():
    def __init__(self,df):
        self.data = None
        self.dataset = None
        self.training_data_len = None
        self.scaler = None
        self.scaled_data = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.cnn_model = None
        self.lstm_model = None
        self.hybrid_model = None
        self.predictions = None
        

    @classmethod
    def create(cls):
        self = cls()
        self.run()
        return self

    # Change this so it can accept the df data
    def load_data(self):
        df = yf.download("MSFT", start="2012-01-01", end="2019-12-17")
        self.df = pd.DataFrame(df)
        self.data = df[['Close']]
        self.dataset = self.data.values
        self.training_data_len = math.ceil(len(self.dataset) * .8)
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_data = self.scaler.fit_transform(self.dataset)

    def preprocess_training_data(self):
        train_data = self.scaled_data[0:self.training_data_len, :]
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, self.y_train = np.array(x_train), np.array(y_train)
        self.x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    def preprocess_testing_data(self):
        test_data = self.scaled_data[self.training_data_len - 60: , :]
        x_test = []
        self.y_test = self.dataset[self.training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        self.x_test = np.array(x_test)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

    def build_cnn_model(self):
        cnn_model = Sequential()
        cnn_model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape = (self.x_train.shape[1], 1)))
        cnn_model.add(MaxPooling1D(pool_size=2))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(60, activation='relu'))
        self.cnn_model = cnn_model

    def build_lstm_model(self):
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        lstm_model.add(LSTM(50, return_sequences=False))
        lstm_model.add(Dense(25))
        self.lstm_model = lstm_model

    def combine_models(self):
        hybrid_model = Sequential()
        hybrid_model.add(self.cnn_model)
        hybrid_model.add(Reshape((60, 1)))
        hybrid_model.add(self.lstm_model)
        hybrid_model.add(Dense(1))
        self.hybrid_model = hybrid_model
        self.hybrid_model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        self.hybrid_model.fit(self.x_train,
                              self.y_train,
                              batch_size=1,
                              epochs=1,
                              validation_data=(self.x_test, self.y_test),
                              callbacks=[early_stop, reduce_lr])

    def evaluate_model(self):
        pass


    def prediction(self):
        predictions = self.hybrid_model.predict(self.x_test)
        self.predictions = self.scaler.inverse_transform(predictions)


    def save_model(self):
        pass


    def load_model(self):
        pass

    def plot_prediction(self):
        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:]
        valid['Predictions'] = self.predictions

        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()


    def run(self):
        self.load_data()
        self.preprocess_training_data()
        self.preprocess_testing_data()
        self.build_cnn_model()
        self.build_lstm_model()
        self.combine_models()
        self.train()
        self.prediction()
        self.plot_prediction()
        self.print_df()

    def print_df(self):
        print(self.df)

    def build(self):
        return super().build()
    
    def load_and_preprocess_data(self):
        return super().load_and_preprocess_data()

    