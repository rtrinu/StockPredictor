import yfinance as yf
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler, RobustScaler

class LstmSignalModel():
    def __init__(self):
        pass

    def create(cls):
        pass

    def load_data(self):
        df = yf.download("MSFT", start="2012-01-01", end="2019-12-17")
        df = pd.DataFrame(df)
        self.data = df[['Close']]
        self.dataset = self.data.values
        self.training_data_len = math.ceil(len(self.dataset) * .8)
        self.minmax_scaler = MinMaxScaler(feature_range=(0,1))
        self.robust_scaler = RobustScaler()