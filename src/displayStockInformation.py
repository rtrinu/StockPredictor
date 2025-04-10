import pandas as pd

def display_info(df):
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    current_price = df['Close'].iloc[-1]
    
    df['Previous_Close'] = pd.to_numeric(df['Previous_Close'], errors='coerce')
    price_change = current_price - df['Previous_Close'].iloc[-1]
    price_percentage_change = (price_change / df['Previous_Close'].iloc[-1]) * 100
    
    open_price = df['Open'].iloc[-1]
    high_price = df['High'].iloc[-1]
    low_price = df['Low'].iloc[-1]
    volume = df['Volume'].iloc[-1]

    stock_data = {
        "current_price": f"{float(current_price):,.2f}", 
        "price_change": f"{float(price_change):,.2f}",
        "price_change_percentage": f"{float(price_percentage_change):.2f}%",  
        "open_price": f"{float(open_price):,.2f}",
        "high_price": f"{float(high_price):,.2f}",
        "low_price": f"{float(low_price):,.2f}",
        "close_price": f"{float(current_price):,.2f}", 
        "volume": f"{int(volume):,.0f}"  
    }
    if stock_data:
        return pd.Series(stock_data)
    else:
        raise ValueError("Stock data is not valid")