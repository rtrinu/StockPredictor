import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import matplotlib
matplotlib.use('TkAgg')
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
        self.lookback = 30
        self.forecast_horizon = 10

    @classmethod
    def create(cls, df, stock_name):
        self = cls(df, stock_name)
        self.load_model()
        self.load_and_preprocess_data()
        if self.hybrid_model is None:
            print(f"No existing model for {self.stock_name}")
            self.run()
            self.save_model()
        self.predict_future_stocks()
        #self.compare_models()
        return self

    def _load_data(self):
        self.data = self.df[['Close']]
        self.dataset = self.data.values
        self.training_data_len = math.ceil(len(self.dataset) * .8)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(self.dataset[:self.training_data_len])
        self.scaled_data = self.scaler.transform(self.dataset)
        print(f"Sample of dates: {self.df.index[:5]}")

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
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
        history = self.hybrid_model.fit(
            self.x_train,
            self.y_train,
            batch_size=20,
            epochs=50,
            validation_data=(self.x_test, self.scaler.transform(self.y_test)),
            callbacks=[early_stop, reduce_lr],
        )

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
            

    @tf.function(reduce_retracing=True)
    def predict_step(self, current_sequence):
        return self.hybrid_model.predict(current_sequence, verbose=0)

    def predict_future(self):
        last_sequence = self.df['Close'].tail(self.lookback).values
        last_sequence = last_sequence.reshape(-1, 1)
        
        last_sequence = self.scaler.transform(last_sequence)
        
        current_sequence = last_sequence.reshape(1, self.lookback, 1)
        
        future_predictions = []
        
        for _ in range(self.forecast_horizon):
            next_pred = self.hybrid_model.predict_step(current_sequence)
            future_predictions.append(next_pred[0, 0])
            
            current_sequence = current_sequence.reshape(self.lookback, 1)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
            current_sequence = current_sequence.reshape(1, self.lookback, 1)
        
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        padded_predictions = np.zeros((len(future_predictions), self.scaled_data.shape[1]))
        padded_predictions[:, 0] = future_predictions[:, 0]
        future_predictions = self.scaler.inverse_transform(padded_predictions)[:, 0]
        
        if isinstance(self.df.index, pd.DatetimeIndex):
            last_date = self.df.index[-1]
        else:
            last_date = pd.Timestamp.today()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=self.forecast_horizon,
            freq='B'
        )

        forecast_df = pd.DataFrame({
            'Predicted_Price': future_predictions
        }, index=future_dates)
        
        return forecast_df
    
    def plot_prediction(self):
        future_pred_df = self.predict_future()
        future = pd.DataFrame(index=future_pred_df.index)
        future["Future_Predictions"] = future_pred_df["Predicted_Price"]
        
        # Create a date index for the training and validation sets
        train_dates = self.df['Date'][:self.training_data_len].values
        valid_dates = self.df['Date'][self.training_data_len:].values
        
        # Create DataFrames for train and valid using the dates
        train = pd.DataFrame(data=self.data[:self.training_data_len].values, index=train_dates, columns=['Close'])
        valid = pd.DataFrame(data=self.data[self.training_data_len:].values, index=valid_dates, columns=['Close'])
        
        # Add predictions to the valid DataFrame
        valid["Predictions"] = self.predictions

        plt.figure(figsize=(16, 8))
        plt.title('Model')
        
        # Plotting the data with dates on the x-axis
        plt.plot(train.index, train['Close'], label='Train')
        plt.plot(valid.index, valid[['Close', 'Predictions']], label='Val')
        plt.plot(future.index, future['Future_Predictions'], linestyle='--', linewidth=2, label='Future Predictions')
        
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        filename = f'{self.stock_name}_hybrid.png'
        filepath = os.path.join('app', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def predict_future_stocks(self):
        self.output_predictions()
        self.predict_future()
        self.plot_prediction()