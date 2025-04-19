import pandas as pd
import math
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

class RandomForest():
    def __init__(self, df, stock_name):
        self.df = df
        self.features = ['Previous_Close', 'RSI', 'EMA_20', 'SMA_20', 'MACD', 'MACD_signal', 'MACD_histogram','Compound Sentiment']
        self.target = 'Close'
        self.model = None
        self.stock_name = stock_name
        self.model_folder = 'models'
        self.forecast_horizon = 10

    @classmethod
    def create(cls, df, stock_name):
        self = cls(df, stock_name)
        self.load_and_process_data()
        self.load_model()
        self.existing_model()
        self.prediction()
        return self

    def load_data(self):
        columns = self.features.copy()
        columns.append(self.target)
        self.data = self.df[columns]
        self.dataset = self.data.values
        self.training_data_len = math.ceil(len(self.dataset) * 0.8)

    def process_data(self):
        if self.data.isnull().values.any():
            print("Missing values found. Handling missing values...")
            self.data.dropna(inplace=True)
        x = self.data[self.features]
        y = self.data[self.target]
        y = y.values.squeeze()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def build(self):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.x_train, self.y_train)
        self.model = grid_search.best_estimator_

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
        if self.predictions is not None:
            mse = mean_squared_error(self.y_test, self.predictions)
            mae = mean_absolute_error(self.y_test, self.predictions)
            r2 = r2_score(self.y_test, self.predictions)

            print("Model Evaluation:")
            print(f"Mean Squared Error: {mse}")
            print(f"Mean Absolute Error: {mae}")
            print(f"R-squared Score: {r2}")
            return mse, mae, r2
        return None, None, None
    
    def predict_step(self, input_data):
        return self.model.predict(input_data)

    def predict_future(self, forecast_horizon=10):
        last_sequence = self.data.iloc[-1].copy()

        rolling_prices = list(self.data['Close'].iloc[-20:])
        ema_20 = self.data['EMA_20'].iloc[-1]
        ema_12 = self.data['MACD'].iloc[-1] + self.data['MACD_signal'].iloc[-1]
        ema_26 = ema_12 - self.data['MACD'].iloc[-1]
        macd_signal = self.data['MACD_signal'].iloc[-1]
        compound_sentiment = self.data['Compound Sentiment'].iloc[-1]

        future_predictions = []

        for _ in range(forecast_horizon):
            input_data = last_sequence[self.features].values.reshape(1, -1)
            predicted_price = self.predict_step(input_data)[0]
            future_predictions.append(predicted_price)

            last_sequence['Previous_Close'] = predicted_price
            last_sequence['Close'] = predicted_price

            rolling_prices.append(predicted_price)
            if len(rolling_prices) > 20:
                rolling_prices.pop(0)
            last_sequence['SMA_20'] = sum(rolling_prices) / len(rolling_prices)

            alpha_ema_20 = 2 / (20 + 1)
            ema_20 = (predicted_price - ema_20) * alpha_ema_20 + ema_20
            last_sequence['EMA_20'] = ema_20

            alpha_ema_12 = 2 / (12 + 1)
            alpha_ema_26 = 2 / (26 + 1)
            ema_12 = (predicted_price - ema_12) * alpha_ema_12 + ema_12
            ema_26 = (predicted_price - ema_26) * alpha_ema_26 + ema_26
            macd = ema_12 - ema_26
            last_sequence['MACD'] = macd
            alpha_macd_signal = 2 / (9 + 1)
            macd_signal = (macd - macd_signal) * alpha_macd_signal + macd_signal
            last_sequence['MACD_signal'] = macd_signal
            last_sequence['MACD_histogram'] = macd - macd_signal

            gain = max(0, predicted_price - last_sequence['Previous_Close'])
            loss = max(0, last_sequence['Previous_Close'] - predicted_price)
            rs = gain / loss if loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            last_sequence['RSI'] = rsi

        if isinstance(self.df.index, pd.DatetimeIndex):
            last_date = self.df.index[-1]
        else:
            last_date = pd.Timestamp.today()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=forecast_horizon, 
            freq='B'
        )
        future_dates.floor('T')

        forecast_df = pd.DataFrame({
            'Predicted_Price': future_predictions
        }, index=future_dates)
        forecast_df.index = forecast_df.index.date
        
        return forecast_df 

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

    def existing_model(self):
        if self.model is None:
            self.run()
            self.save_model()
    
