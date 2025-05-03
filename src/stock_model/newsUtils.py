import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime as dt, timedelta
from dotenv import load_dotenv
from newsapi import newsapi_client
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv('NEWS_API_KEY')

class StockNews:
    def __init__(self, symbol: str) -> None:
        """
        Initializes the StockNewsAnalyzer class.

        :param news_api_key: str - Your NewsAPI key.
        """
        self.configure()
        self.newsapi = newsapi_client.NewsApiClient(api_key=NEWSAPI_KEY)
        self.sia = SentimentIntensityAnalyzer()
        self.symbol = symbol
        self.df = None
        self.initialise()
        
   
    

    def configure(self) -> None:
        """
        Configure the environment by loading environment variables from a .env file.
        """
        load_dotenv()

    def news_fetch(self) -> None:
        """
        Fetch the latest stock news articles for a given stock symbol from NewsAPI and Google News.

        :param symbol: str - The stock symbol (e.g., 'AAPL' for Apple)
        """
        if not self.symbol:
            print("Error, Stock symbol missing")
            return
        if not self.newsapi:
            print("No api works")
            return
        end_date = dt.today()
        start_date = end_date - timedelta(days=25)

        newsapi_response = self.newsapi.get_everything(
            q=self.symbol,
            from_param=start_date,
            to=end_date,
            language='en',
            sort_by='publishedAt',
            page_size=100
        )

        news_data = []


        if newsapi_response.get('articles'):
            for article in newsapi_response['articles']:
                title = article['title']
                published_at = dt.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d')
                news_data.append({
                    'Title': title,
                    'Date': published_at
                })

        google_news_url = f'https://news.google.com/rss/search?q={self.symbol}+stocks'
        google_news_response = requests.get(google_news_url)
        soup = BeautifulSoup(google_news_response.content, 'lxml-xml')
        items = soup.find_all('item')

        for item in items:
            title = item.title.text
            published_at = dt.strptime(item.pubDate.text, "%a, %d %b %Y %H:%M:%S %Z").strftime('%Y-%m-%d')
            news_data.append({
                'Title': title,
                'Date': published_at
            })

        #news_data_sorted = sorted(news_data, key=lambda x: x['Date'], reverse=True)
        #self.df = self.df.iloc[::-1].reset_index(drop=True)
        news_data = news_data[::-1]
        self.df = pd.DataFrame(news_data)
        self.df = self.df.sort_values('Date')
        self.df.reset_index(drop=True)
        return self.df


    def vaderpreprocess_text(self) -> None:
        """
        Process the text data from the CSV file, analyze sentiment using VADER, and add a column for compound sentiment scores.

        :param csv_file: str - The name of the CSV file (e.g., 'AAPL_stock_news.csv').
        """
        df = self.df

        res = []

        for i, row in df.iterrows():
            text = row["Title"]
            sentiment = self.sia.polarity_scores(text)
            res.append(sentiment['compound'])

        df['Compound Sentiment'] = res

        self.df = df
        return self.df
    
    def initialise(self):
        self.news_fetch()
        self.vaderpreprocess_text()
