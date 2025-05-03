import pandas_ta as ta
import pandas as pd

class TechnicalIndicatorUtil:

    @staticmethod
    def add_technical_indicators(df):
        if df is None:
            print("No data in dataframe to create indicators")
            return 
        
        columns = df.columns
        for cols in columns:
            df[cols] = pd.to_numeric(df[cols], errors='coerce')
        
        df['Previous_Close'] = df['Close'].shift(1)
        df['Close_Shifted'] = df['Close'].shift(1)
        df['Open_Shifted'] = df['Open'].shift(1)
        df['High_Shifted'] = df['High'].shift(1)
        df['Low_Shifted'] = df['Low'].shift(1)

        df['RSI'] = ta.rsi(df['Close_Shifted'], length=14)
        df['EMA_20'] = ta.ema(df['Close_Shifted'], length=20)
        df['SMA_20'] = ta.sma(df['Close_Shifted'], length=20)
        
        macd= ta.macd(df['Close_Shifted'], length=14)

        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_histogram'] = macd['MACDh_12_26_9']

        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        return df
    
    def generate_technical_signals(df):
        if df is None:
            print("No data available to generate signals")
            return
        df['Signal'] = 0
        trend_signals = {
            'EMA_CROSS': (df['EMA_20'] > df['SMA_20']).astype(int),
            'MACD_CROSS': (df['MACD'] > df['MACD_signal']).astype(int),
            'PRICE_VS_EMA': (df['Close'] > df['EMA_20']).astype(int)
            }
        rsi_signal = pd.Series(0, index=df.index)
        rsi_signal[df['RSI'] < 30] = 1
        rsi_signal[df['RSI'] > 70] = -1
        
        def calculate_signal_strength():
            strength = 0
            strength += trend_signals['EMA_CROSS'] * 2
            strength += trend_signals['MACD_CROSS'] * 2
            strength += trend_signals['PRICE_VS_EMA']
            strength += rsi_signal
            return strength
        
        signal_strength = calculate_signal_strength()
        df.loc[signal_strength >= 2, 'Signal'] = 1
        df.loc[signal_strength <= -2, 'Signal'] = -1
        df.reset_index(inplace=True)
        return df