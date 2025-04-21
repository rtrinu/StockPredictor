from src.stock_model.stockDataUtils import StockDataUtils
from src.stock_model.technicalIndicatorUtils import TechnicalIndicatorUtil
from src.stock_model.newsUtils import StockNews
from src.stock_model.ai_models.lstmcnnHybrid import CnnLSTMHybrid
from src.stock_model.ai_models.randomforest import RandomForest
from src.stock_model.ai_models.decisiontree import DecisionTreeModel
from src.stock_model.ai_models.stacking import StackedModel
from src.displayStockInformation import display_info, display_plot, display_predictions
import pandas as pd
from src.stock_model.stockPrediction import simple_averages, weighted_averages
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
        self._gather_data()
        self._add_technical_indicators()
        self._add_technical_signals()
        self._get_news_articles()
        return self
        #self.create_ai_training_df()
        #self.train_ai_models()
        #self.print_df()
    
    def stock_dictionary(self):
        self.stock_data_utils.fetch_stock_data()

    def _gather_data(self):
        self.stock_data_utils.fetch_stock_data()
        self.stock_symbol = self.stock_data_utils.stock_symbol
        self.stock_name = self.stock_data_utils.stock_name
        self.df = self.stock_data_utils.df
        if self.df is None:
            print("Failed to gather data")
        self.df = pd.DataFrame(self.df)

    def _add_technical_indicators(self):
        if self.df is None:
            print("Dataframe is empty/not loaded to add technical indicators.")
            return
        else:
            self.df = TechnicalIndicatorUtil.add_technical_indicators(self.df)
    
    def _add_technical_signals(self):
        if self.df is None:
            print("Dataframe is empty/not loaded to add technical signals.")
            return
        else:
            self.df = TechnicalIndicatorUtil.generate_technical_signals(self.df)

    def _get_news_articles(self):
        self.news_df = StockNews(self.stock_name).df

    def _create_ai_training_df(self):
        num_news_entries = len(self.news_df['Compound Sentiment'])
        self.news_df = self.news_df.sort_values('Date').reset_index(drop=True)
        if 'Compound Sentiment' not in self.df.columns:
            self.df['Compound Sentiment'] = pd.NA
        self.df['Compound Sentiment'].iloc[-num_news_entries:] = self.news_df['Compound Sentiment'].iloc[-num_news_entries:]
        self.df.drop(['index'], axis=1)
        self.df.to_csv('ai_table.csv')
 
    def _train_ai_models(self):
        print("Training AI...")
        
        self.hybrid = CnnLSTMHybrid.create(self.df, self.stock_name)
        
        #hybrid.plot_prediction()

        self.random_forest = RandomForest.create(self.df, self.stock_name)
        
        self.stacked = StackedModel.create(self.df,self.stock_name)
        
        self.decision = DecisionTreeModel.create(self.df,self.stock_name)

    def output_predictions(self):
        hybrid_predictions = self.hybrid.predict_future()
        rf_predictions = self.random_forest.predict_future()
        stacked_predictions = self.stacked.predict_future()
        decision_predictions = self.decision.predict_future()
        simple = simple_averages(rf_predictions['Predicted_Price'], hybrid_predictions['Predicted_Price']).tolist()
        weighted = weighted_averages(rf_predictions['Predicted_Price'], hybrid_predictions['Predicted_Price']).tolist()
       
        numerical_models = {
            'Hybrid model':hybrid_predictions.to_dict(orient='records'),
            'Random Forest Model':rf_predictions.to_dict(orient='records')
        }
        averages = {
            'Simple':simple,
            'Weighted':weighted
            }
        signals = {
            'Stacked Model':stacked_predictions.to_dict(orient='records'),
            'Decision Tree':decision_predictions.to_dict(orient='records')
        }
        return {
        'numerical_models': numerical_models,
        'averages': averages,
        'signals': signals
        }

        
    def print_df(self):
        print(self.df)
        print(self.news_df)

    def display_information(self):
        return display_info(self.df, self.stock_name,self.stock_symbol)
    
    def display_plot(self):
        return display_plot(self.df)
    
    def display_prediction(self):
        predictions = self.output_predictions()
        return display_predictions(predictions)
    
    def return_stock_symbol(self):
        return self.stock_symbol
    
    def create_and_train(self):
        self._create_ai_training_df()
        self._train_ai_models()
        