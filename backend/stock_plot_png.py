import yfinance as yf
import matplotlib.pyplot as plt
import random as rnd
from datetime import datetime as dt, timedelta
import pandas as pd
import os

def plot_close_data():
    companies = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "BRK.B",
                  "JNJ", "V", "PG", "JPM", "UNH", "MA", "DIS", "HD", "PFE", "KO", "INTC"]
    end_date = dt.today()
    start_date = end_date - timedelta(days=90)

    first_random_int = rnd.randint(0, 6)
    first_company = companies[first_random_int]

    del companies[first_random_int]

    second_random_int = rnd.randint(0, 5)
    second_company = companies[second_random_int]

    first_company_data = yf.download(first_company, start=start_date, end=end_date)
    second_company_data = yf.download(second_company, start=start_date, end=end_date)

    if first_company_data.empty or second_company_data.empty:
        raise ValueError(f"Data for {first_company} or {second_company} is empty.")

    first_company_data = first_company_data.reset_index()
    second_company_data = second_company_data.reset_index()

    fc_prices = first_company_data['Close']
    sc_prices = second_company_data['Close']

    dates = pd.to_datetime(first_company_data['Date'], errors="coerce")

    if dates.isnull().all():
        raise ValueError("No valid dates available in the data after coercion.")
    
    first_date = dates.min()
    last_date = dates.max()

    if pd.isna(first_date) or pd.isna(last_date):
        raise ValueError("Invalid date range: Dates are NaN or Inf.")

    plt.figure(figsize=(20, 12),dpi=300)
    plt.xlim(first_date, last_date)

    plt.plot(dates, fc_prices, label=f'{first_company}', color='blue')
    plt.plot(dates, sc_prices, label=f'{second_company}', color='green')

    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'Closing Prices of {first_company} and {second_company} ')
    
    plt.legend()
    plt.grid(True)
    
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    filename = 'static.png'
    filepath = os.path.join('app', filename)
    plt.savefig(filepath)
    plt.close()  

    return filename, filepath
