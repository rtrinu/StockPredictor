from src.stock_model.stockDataUtils import StockDataUtils
from src.stock_model.technicalIndicatorUtils import TechnicalIndicatorUtil
from src.stock_model.newsUtils import StockNews
from src.stock_model.ai_models.lstmcnnHybrid import CnnLSTMHybrid
from src.stock_model.ai_models.randomforest import RandomForestModel
import pandas as pd
class Stock:
    def __init__(self, stock_symbol:str):
        self.stock_symbol = stock_symbol
        self.stock_name = None
        self.df = None
        self.news_df = None
        self.stock_data_utils = StockDataUtils(self.stock_symbol)

    @classmethod
    def create(cls,stock_symbol):
        self = cls(stock_symbol)
        self.gather_data()
        self.add_technical_indicators()
        self.add_technical_signals()
        self.get_news_articles()
        self.create_ai_training_df()
        self.train_ai_models()
        #self.print_df()

    def gather_data(self):
        self.stock_data_utils.fetch_stock_data()
        self.stock_name = self.stock_data_utils.stock_name
        self.df = self.stock_data_utils.df
        if self.df is None:
            print("Failed to gather data")
        self.df = pd.DataFrame(self.df)

    def add_technical_indicators(self):
        if self.df is None:
            print("Dataframe is empty/not loaded to add technical indicators.")
            return
        else:
            self.df = TechnicalIndicatorUtil.add_technical_indicators(self.df)
    
    def add_technical_signals(self):
        if self.df is None:
            print("Dataframe is empty/not loaded to add technical signals.")
            return
        else:
            self.df = TechnicalIndicatorUtil.generate_technical_signals(self.df)

    def get_news_articles(self):
        self.news_df = StockNews(self.stock_name).df

    def create_ai_training_df(self):
        self.news_df = self.news_df.sort_values('Date').reset_index(drop=True)
        self.df['Compound Sentiment'] = self.news_df['Compound Sentiment']
        self.df.drop(['index'], axis=1)

    def train_ai_models(self):
        print("Training AI...")
        hybrid = CnnLSTMHybrid.create(self.df, self.stock_name)
        #random_forest = RandomForestModel(self.df)
        
        
    def print_df(self):
        print(self.df)
        print(self.news_df)