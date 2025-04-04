import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Flatten,Reshape, Conv1D, MaxPooling1D, LSTM, Dense #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('fivethirtyeight')


class CnnLSTMHybrid():
    def __init__(self, df, stock_name):
        self.df = df
        self.stock_name = stock_name
        self.model_folder = "models"
        self.data = None
        self.dataset = None
        self.training_data_len = None
        self.scaler = None
        self.scaled_data = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.hybrid_model = None
        self.predictions = None

    @classmethod
    def create(cls, df, stock_name):
        self = cls(df, stock_name)
        self.load_model()
        self.load_and_preprocess_data()
        if self.hybrid_model is None:
            self.run()
        self.compare_models()
        return self

    def _load_data(self):
        self.data = self.df[['Close']]
        self.dataset = self.data.values
        self.training_data_len = math.ceil(len(self.dataset) * .8)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(self.dataset[:self.training_data_len])
        self.scaled_data = self.scaler.transform(self.dataset)

    def _preprocess_training_data(self):
        train_data = self.scaled_data[:self.training_data_len]
        x_train, y_train = [], []
        for i in range(30, len(train_data)):
            x_train.append(train_data[i - 30:i])
            y_train.append(train_data[i, 0])
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def _preprocess_testing_data(self):
        test_data = self.scaled_data[self.training_data_len - 30:]
        x_test = []
        for i in range(30, len(test_data)):
            x_test.append(test_data[i - 30:i])
        self.x_test = np.array(x_test)
        self.y_test = self.dataset[self.training_data_len:]

    def _build_model(self):
        cnn_model = Sequential()
        cnn_model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(self.x_train.shape[1], 1)))
        cnn_model.add(MaxPooling1D(pool_size=2))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(30, activation='relu'))
        
        self.cnn_model = cnn_model

        lstm_model = Sequential()
        lstm_model.add(LSTM(100, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        lstm_model.add(LSTM(50, return_sequences=False))
        lstm_model.add(Dense(30))
        
        self.lstm_model = lstm_model

        hybrid_model = Sequential()
        hybrid_model.add(self.cnn_model)
        hybrid_model.add(Reshape((30, 1)))
        hybrid_model.add(self.lstm_model)
        hybrid_model.add(Dense(1)) 
        self.hybrid_model = hybrid_model
        self.hybrid_model.compile(optimizer='adam', loss='mean_squared_error')

    def _train(self):
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
        history = self.hybrid_model.fit(
            self.x_train,
            self.y_train,
            batch_size=20,
            epochs=50,
            validation_data=(self.x_test, self.scaler.transform(self.y_test)),
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.show()

    def baseline_mae(self):
        mean_price = np.mean(self.dataset[:self.training_data_len])
        naive_preds = np.full_like(self.y_test, mean_price)
        mae = mean_absolute_error(self.y_test, naive_preds)
        print(f"Baseline MAE (predicting mean): {mae:.4f}")

    def evaluate_model(self):
        self.output_predictions()
        if self.predictions is None:
            return None, None, None
        actual = self.y_test
        predicted = self.predictions
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        self.baseline_mae()
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared Score: {r2:.4f}")
        return mse, mae, r2

    def _prediction(self):
        if self.hybrid_model is not None:
            return self.hybrid_model.predict(self.x_test, verbose=0)
        return None

    def output_predictions(self):
        predictions = self._prediction()
        if predictions is not None:
            padded = np.zeros((len(predictions), self.scaled_data.shape[1]))
            padded[:, 0] = predictions[:, 0]
            self.predictions = self.scaler.inverse_transform(padded)[:, 0]
        else:
            self.predictions = None

    def save_model(self):
        os.makedirs(self.model_folder, exist_ok=True)
        model_file = os.path.join(self.model_folder, f"{self.stock_name}_CNNLSTMModel_pickle")
        with open(model_file, 'wb') as f:
            pickle.dump(self.hybrid_model, f)

    def load_model(self):
        model_file = os.path.join(self.model_folder, f"{self.stock_name}_CNNLSTMModel_pickle")
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.hybrid_model = pickle.load(f)
        else:
            self.hybrid_model = None

    def plot_prediction(self):
        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:].copy()
        valid["Predictions"] = self.predictions
        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()

    def load_and_preprocess_data(self):
        self._load_data()
        self._preprocess_training_data()
        self._preprocess_testing_data()

    def run(self):
        self._build_model()
        self._train()

    def compare_models(self):
        existing_mse, existing_mae, existing_r2 = self.evaluate_model()
        self.run()
        new_mse, new_mae, new_r2 = self.evaluate_model()
        
        better = sum([new_mse < existing_mse, new_mae < existing_mae, new_r2 > existing_r2]) >= 2
        
        if better:
            print("New model performs better. Saving the new model.")
            self.save_model()
        else:
            print("Using the existing model.")
            self.load_model()
            self.output_predictions()
            
        self.plot_prediction()
