import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

class StackedModel():
    def __init__(self,df):
        self.df = df
        
    @classmethod
    def create(cls,df):
        self = cls(df)
        self.load_data()
        self.build_models()
        self.train()
        self.predict()
        self.evaluate()
        return self

    def load_data(self):
        self.df = self.df.dropna()
        x = self.df.drop(columns=['Signal', 'Date']) 
        y = self.df['Signal']
        self.x_train,self.x_test, self.y_train,self.y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    def build_models(self):
        base_models = [('rf', RandomForestRegressor(n_estimators=100)),
                       ('svc',SVR())]
        meta_model = LinearRegression()
        stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
        self.model = stacking_model
    
    def train(self):
        self.model.fit(self.x_train,self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.x_test)
        return y_pred
    
    def evaluate(self):
        y_pred = self.predict()
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"Mean Squared Error of Stacked Model: {mse:.2f}")
        print(f"R-squared Score of Stacked Model: {r2:.2f}")

    def predict_future(self,days=10):
        last_data_point = self.df.iloc[-1].drop(['Signal', 'Date'])
        future_predictions = []
        for _ in range(days):
            next_prediction = self.model.predict(last_data_point.values.reshape(1, -1))[0]
            if 0.5 > next_prediction > -0.5:
                next_prediction = 0
            elif next_prediction >= 0.5:
                next_prediction = 1
            else:
                next_prediction = -1 
            future_predictions.append(next_prediction)
            last_data_point = self.df.iloc[-1].drop(['Signal', 'Date'])
        
        if isinstance(self.df.index, pd.DatetimeIndex):
            last_date = self.df.index[-1]
        else:
            last_date = pd.Timestamp.today()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=days, 
            freq='B'
        )
        future_dates.floor('T')
        forecast_df = pd.DataFrame({
            'Signal': future_predictions
        }, index=future_dates)
        forecast_df.index.name = 'Date'
        forecast_df.index = forecast_df.index.date
        return forecast_df