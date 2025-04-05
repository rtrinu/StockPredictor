import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

class DualScalerPreprocessor():
    def __init__(self):
        self.minmax_features = ['Close', 'High', 'Low', 'Open', 'Previous_Close', 'Close_Shifted']
        self.robust_features = ['Volume', 'RSI', 'EMA_20', 'SMA_20', 'MACD', 
                              'MACD_signal', 'MACD_histogram', 'Compound Sentiment']
        self.minmax_scaler = MinMaxScaler(feature_range=(0,1))
        self.robust_scaler = RobustScaler()

    def fit(self,df):
        self.minmax_scaler.fit(df[self.minmax_features])
        self.robust_scaler.fit(df[self.robust_features])
    
    def transform(self, df):
        scaled_minmax = self.minmax_scaler.transform(df[self.minmax_features])
        scaled_robust = self.robust_scaler.transform(df[self.robust_features])

        scaled_data = np.hstack((scaled_minmax, scaled_robust))
        return scaled_data

    def fit_transform(self,df):
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self):
        minmax_data = self.scaled_data[:, :len(self.minmax_features)]
        robust_data = self.scaled_data[:, len(self.minmax_features):]

        original_minmax = self.minmax_scaler.inverse_transform(minmax_data)
        original_robust = self.robust_scaler.inverse_transform(robust_data)

        return np.hstack((original_minmax, original_robust))