from src.stock_model.dictionary import fetch_sp500_data, get_stock_symbol_from_name
import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
class StockDataUtils:
    def __init__(self, stock_symbol:str):
        self.stock_dict = fetch_sp500_data()
        self.stock_symbol = stock_symbol
        self.stock_name = None
        self.df = None

    def user_stock_validation(self):
        user_stock = self.stock_symbol.upper()
        self.stock_symbol, self.stock_name = get_stock_symbol_from_name(user_stock, self.stock_dict)
        if not self.stock_name:
            raise Exception(f"Stock symbol or name '{self.user_stock}' not found in S&P 500.")
        return self.stock_symbol, self.stock_name
    
    def fetch_stock_data(self):
        end_date = datetime.date.today()
        start_date = end_date - relativedelta(months=5)
        self.user_stock_validation()
        print(f"Fetching data for '{self.stock_name}'...")
        self.df = yf.download(self.stock_symbol, start=start_date, end=end_date)
        if self.df.empty:
            print(f"No data found for {self.stock_name}")
            return None
        self.df.columns = self.df.columns.droplevel(1)
        return self.df