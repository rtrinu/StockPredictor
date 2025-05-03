import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import os
import pickle
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

class DecisionTreeModel():
    def __init__(self,df,stock_name):
        self.df = df
        self.stock_name = stock_name
        self.features = ['Close','High','Low','Open','Previous_Close','RSI','EMA_20','SMA_20','MACD','MACD_signal','MACD_histogram']
        self.target = 'Signal'
        self.lookback = 1
        self.forecast_horizon = 10
        self.model_folder = 'models'
        self.model = None

    @classmethod
    def create(cls,df,stock_name):
        self = cls(df,stock_name)
        self.load_model()
        self.process_data()
        self.existing_model()
        self.predict()
        self.evaluate()
        return self

    def process_data(self):
        x = self.df[self.features]
        y = self.df[self.target]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.2, random_state=42)

    def build(self):
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10]
        }
        clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
        self.model = clf.fit(self.x_train,self.y_train)
        return self.model

    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    def evaluate(self):
        f1 = f1_score(self.y_test, self.y_pred)
        print(f"F1-Scores: {f1}")

    def predict_future(self):
            last_sequence = self.df[self.features].tail(self.lookback).values     
            current_sequence = last_sequence.reshape(1, self.lookback, len(self.features))
            future_predictions = []

            for _ in range(self.forecast_horizon):
                next_pred = self.model.predict(current_sequence.reshape(1, -1))
                future_predictions.append(next_pred[0])
                
                current_sequence = current_sequence.reshape(self.lookback, len(self.features))
                current_sequence = np.roll(current_sequence, -1, axis=0)  
                current_sequence[-1] = next_pred  
                current_sequence = current_sequence.reshape(1, self.lookback, len(self.features))  

            future_predictions = np.array(future_predictions).reshape(-1, 1)
            
            if isinstance(self.df.index, pd.DatetimeIndex):
                last_date = self.df.index[-1]
            else:
                last_date = pd.Timestamp.today()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=self.forecast_horizon,
                freq='B' 
            )
            future_dates.floor('T')
            
            forecast_df = pd.DataFrame({
                'Signal': future_predictions.flatten()
            }, index=future_dates)
            forecast_df.index = forecast_df.index.date
            
            return forecast_df
    
    def save_model(self):
        os.makedirs(self.model_folder, exist_ok=True)
        model_file = os.path.join(self.model_folder, f"{self.stock_name}_decision_pickle")
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        model_file = os.path.join(self.model_folder, f"{self.stock_name}_decision_pickle")
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = None

    def existing_model(self):
        if self.model is None:
            self.build()
            self.save_model()

    