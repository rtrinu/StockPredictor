import math
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RandomForestModel():
    def __init__(self, df):
        self.feature_columns = ['Previous Close', 'Open', 'High', 'Low', 'Volume']
        self.target_column = 'Close'

    @classmethod
    def create(cls):
        self = cls()
        self.run()
        return self

    #Change this so it can accept df data
    def load_data(self):
        df = yf.download("MSFT", start="2012-01-01", end="2019-12-17")
        df['Previous Close'] = df['Close'].shift(1)
        df.dropna(inplace=True)
        self.data = df[self.feature_columns + [self.target_column]]
        self.dataset = self.data.values
        self.training_data_len = math.ceil(len(self.dataset) * 0.8)

    def process_data(self):
        x = self.data[self.feature_columns]
        y = self.data[self.target_column]
        y = y.values.squeeze()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y,test_size=0.2, random_state=42)
    
    def build(self):
        self.model = RandomForestRegressor(n_estimators=100,max_depth=10,min_samples_split=2,min_samples_leaf = 1,random_state=42)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def prediction(self):
        self.predictions = self.model.predict(self.x_test)

    def evaluate_model(self):
        mse = mean_squared_error(self.y_test, self.predictions)
        mae = mean_absolute_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)

        print("Model Evaluation:")
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared Score: {r2}")

    def plot_predictions(self):
        results = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': self.predictions
        })
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(results['Actual'], label='Actual')
        plt.plot(results['Predicted'], label='Predicted', alpha=0.7)
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def run(self):
        self.load_data()
        self.process_data()
        self.build()
        self.train()
        self.prediction()
        self.evaluate_model()
        self.plot_predictions()

        