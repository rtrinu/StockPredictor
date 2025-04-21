import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np

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
    
def display_plot(df):
    df['Date'] = pd.to_datetime(df['Date'])
    today = datetime.today()
    start_date = today - timedelta(days=90)
    recent_df = df[df['Date']>=start_date]
    close_prices = recent_df['Close']
    dates = recent_df['Date']
    plt.figure(figsize=(16,8))
    plt.plot(dates,close_prices)
    plt.title('Stock Closing Prices - Last 90 Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.grid(False)
    plt.tight_layout()
    filename = 'static.png'
    filepath = os.path.join('static', filename)
    plt.savefig(filepath)
    plt.close() 
    return filename, filepath

def display_predictions(predictions_dict):
    hybrid_prices = [item['Predicted_Price'] for item in predictions_dict['numerical_models']['Hybrid model']]
    rf_prices = [item['Predicted_Price'] for item in predictions_dict['numerical_models']['Random Forest Model']]
    simple_averages = predictions_dict['averages']['Simple']
    weighted_averages = predictions_dict['averages']['Weighted']
    x = range(len(hybrid_prices))

    plt.figure(figsize=(12, 6))
    plt.plot(x, hybrid_prices, label='Hybrid Model', linewidth=2)
    plt.plot(x, simple_averages, label='Simple Average', linewidth=2)
    plt.plot(x, weighted_averages, label='Weighted Average', linewidth=2)

    plt.title('Price Predictions with Signals', fontsize=14)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')

    min_price = min(min(hybrid_prices), min(rf_prices), min(simple_averages), min(weighted_averages))
    max_price = max(max(hybrid_prices), max(rf_prices), max(simple_averages), max(weighted_averages))
    plt.ylim(min_price - 0.5, max_price + 0.5)

    if not os.path.exists('static'):
        os.makedirs('static')

    plt.savefig('static/prediction.png', dpi=300, bbox_inches='tight')

    plt.close()