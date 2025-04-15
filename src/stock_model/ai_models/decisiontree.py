import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

class DecisionTreeModel():
    def __init__(self,df):
        self.df = df
        self.features = ['Close','High','Low','Open','Previous_Close','RSI','EMA_20','SMA_20','MACD','MACD_signal','MACD_histogram']
        self.target = 'Signal'
        self.lookback = 1
        self.forecast_horizon = 10

    @classmethod
    def create(cls,df):
        self = cls(df)
        self.process_data()
        self.build()
        self.predict()
        self.evaluate()
        return self

    def process_data(self):
        x = self.df[self.features]
        y = self.df[self.target]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.2, random_state=42)

    def build(self):
        clf = DecisionTreeClassifier()
        self.model = clf.fit(self.x_train,self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    def evaluate(self):
        dict = {"Actual":self.y_test,"Predicted":self.y_pred}
        print("Accuracy:", metrics.accuracy_score(self.y_test, self.y_pred))

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
                'Predicted_Signal': future_predictions.flatten()
            }, index=future_dates)
            forecast_df.index = forecast_df.index.date
            
            return forecast_df