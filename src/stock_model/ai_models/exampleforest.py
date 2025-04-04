import yfinance as yf
import pandas as pd
import numpy as np
import math
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from xgboost as xgb

class randomForest():
    def __init__(self,df, stock_name):
        self.df = df
        self.features = ['Previous_Close','RSI','EMA_20','SMA_20','MACD','MACD_signal','MACD_histogram']
        self.target = 'Close'
        self.model = None
        self.stock_name = stock_name
        self.model_folder = 'models'

    @classmethod
    def create(cls,df, stock_name):
        self = cls(df, stock_name)
        self.load_and_process_data()
        self.load_model()
        if self.model is None:
            self.run()
            self.save_model()
        self.compare_models()
        

    def load_data(self):
        columns = self.features.copy()
        columns.append(self.target)
        self.data = self.df[columns]
        self.dataset = self.data.values
        self.training_data_len = math.ceil(len(self.dataset) * 0.8)
        
    def process_data(self):
        x = self.data[self.features]
        y = self.data[self.target]
        y = y.values.squeeze()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y,test_size=0.2, random_state=42)
    
    def build(self):
        self.model = RandomForestRegressor(n_estimators=100,max_depth=10,min_samples_split=2,min_samples_leaf = 1,random_state=42)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def prediction(self):
        if self.model is not None:
            self.predictions = self.model.predict(self.x_test)
            return self.predictions
        print("No model loaded to make predictions")
        return None
    def evaluate_model(self):
        self.prediction()
        if self.prediction() is not None:
            mse = mean_squared_error(self.y_test, self.predictions)
            mae = mean_absolute_error(self.y_test, self.predictions)
            r2 = r2_score(self.y_test, self.predictions)

            print("Model Evaluation:")
            print(f"Mean Squared Error: {mse}")
            print(f"Mean Absolute Error: {mae}")
            print(f"R-squared Score: {r2}")
            return mse, mae, r2
        return None, None, None

    def plot_prediction(self):
        results = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': self.predictions
        })
        plt.figure(figsize=(12, 6))
        plt.plot(results['Actual'], label='Actual')
        plt.plot(results['Predicted'], label='Predicted', alpha=0.7)
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        

    def save_model(self):
        os.makedirs(self.model_folder, exist_ok=True)
        model_file = os.path.join(self.model_folder, f"{self.stock_name}_randomforest_pickle")
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        model_file = os.path.join(self.model_folder, f"{self.stock_name}_randomforest_pickle")
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = None

    def load_and_process_data(self):
        self.load_data()
        self.process_data()

    def run(self):
        self.build()
        self.train()
    
    def predict_model(self):
        self.prediction()
        self.evaluate_model()
        

    def compare_models(self):
        existing_mse, existing_mae, existing_r2 = self.evaluate_model()
        self.run()
        new_mse, new_mae, new_r2 = self.evaluate_model()
        print(existing_mae, existing_mse, existing_r2, new_mae, new_mse, new_r2)
        
        better = sum([new_mse < existing_mse, new_mae < existing_mae, new_r2 > existing_r2]) >= 2
        
        if better:
            print("New model performs better. Saving the new model.")
            self.save_model()
        else:
            print("Using the existing model.")
            self.load_model()
            self.prediction()
            
        self.plot_prediction()