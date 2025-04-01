import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import BatchNormalization,Input,Reshape, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.stock_model.ai_models.base_model import AIModel
plt.style.use('fivethirtyeight')

class CnnLSTMHybrid():
    def __init__(self,df, stock_name):
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
        self.df = df
        self.stock_name = stock_name
        self.model_folder = "models"
        

    @classmethod
    def create(cls,df, stock_name):
        self = cls(df, stock_name)
        self.load_model()            
        self.load_data()
        self.preprocess_training_data()
        self.preprocess_testing_data()
        if self.hybrid_model is None:
            self.run()
        existing_mse, existing_mae, existing_r2 = self.evaluate_model()
        self.run()
        self.prediction()
        new_mse, new_mae, new_r2 = self.evaluate_model()
        better_metrics_count = 0
        if new_mse < existing_mse:
            better_metrics_count += 1
        if new_mae < existing_mae:
            better_metrics_count += 1
        if new_r2 > existing_r2:  
            better_metrics_count += 1
        if better_metrics_count >= 2:
            print("Newer model is better. Saving the newer model.")
            self.save_model()
        else:
            print("Using the existing model.")
            self.hybrid_model = self.load_model()
        self.plot_prediction()
        
        return self

    def load_data(self):
        self.data = self.df[['Close']]
        self.dataset = self.data.values
        self.training_data_len = math.ceil(len(self.dataset) * .8)
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_data = self.scaler.fit_transform(self.dataset)

    def preprocess_training_data(self):
        train_data = self.scaled_data[0:self.training_data_len, :]
        x_train = []
        y_train = []
        num_features = train_data.shape[1]
        for i in range(30, len(train_data)):
            x_train.append(train_data[i-30:i, :])
            y_train.append(train_data[i, 0])

        x_train, self.y_train = np.array(x_train), np.array(y_train)
        self.x_train = np.reshape(x_train, (x_train.shape[0], 30, num_features))

    def preprocess_testing_data(self):
        test_data = self.scaled_data[self.training_data_len - 30: , :]
        x_test = []
        num_features = test_data.shape[1]
        self.y_test = self.dataset[self.training_data_len:, :]
        for i in range(30, len(test_data)):
            x_test.append(test_data[i-30:i, :])
        self.x_test = np.array(x_test)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], 30, num_features))

    def build_model(self):
        self.hybrid_model = Sequential([
            Input(shape=(self.x_train.shape[1], self.x_train.shape[2])),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            BatchNormalization(),  
            MaxPooling1D(pool_size=2),
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        self.hybrid_model.compile(optimizer='adam', loss='mean_squared_error')
        self.hybrid_model.summary()


    def train(self):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        self.hybrid_model.fit(self.x_train,
                              self.y_train,
                              batch_size=20,
                              epochs=50,
                              validation_data=(self.x_test, self.y_test),
                              callbacks=[early_stop, reduce_lr])

    def evaluate_model(self):
        self.output_predictions()
        if self.output_predictions() is None:
            print("No predictions available for evaluation")
            return None, None, None
        actual = self.scaler.inverse_transform(self.y_test.reshape(-1,1))
        mse = mean_squared_error(actual, self.predictions)
        mae = mean_absolute_error(actual, self.predictions)
        r2 = r2_score(actual, self.predictions)

        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared Score: {r2:.4f}")

        return mse, mae, r2



    def prediction(self):
        if self.hybrid_model is not None:
            predictions = self.hybrid_model.predict(self.x_test)
            return predictions
        else:
            print("Model is not loaded. Cannot make predictions")
            return None
    
    def output_predictions(self):
        predictions = self.prediction()
        if predictions is not None:
            self.predictions = self.scaler.inverse_transform(predictions)
            return self.predictions
        else:
            self.predictions = None
        


    def save_model(self):
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            print(f"Model folder created.")
        model_file = os.path.join(self.model_folder, f"{self.stock_name}_CNNLSTMModel_pickle")
        
        with open(model_file, 'wb') as f:
            pickle.dump(self.hybrid_model, f)
            print(f"Model saved.")



    def load_model(self):
        model_file = os.path.join(self.model_folder, f"{self.stock_name}_CNNLSTMModel_pickle")
        try:
            with open(model_file, 'rb') as f:
                self.hybrid_model = pickle.load(f)
                print(f"Model loaded successfully")
        except FileNotFoundError:
            print(f"Model file {model_file} not found. Proceeding to train a new model.")
            self.hybrid_model = None
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")

    def plot_prediction(self):
        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:].copy()
        valid.loc[:,"Predictions"] = self.predictions

        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()


    def run(self):
        self.build_model()
        self.train()
        

    def print_df(self):
        print(self.df)


    