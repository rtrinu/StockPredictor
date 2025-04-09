import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

class DecisionTreeModel():
    def __init__(self,df):
        self.df = df
        self.features = ['Close','High','Low','Open','Previous_Close','RSI','EMA_20','SMA_20','MACD','MACD_signal','MACD_histogram']
        self.target = 'Signal'

    @classmethod
    def create(cls,df):
        self = cls(df)
        self.process_data()
        self.build()
        self.predict()
        self.evaluate()

    def process_data(self):
        x = self.df[self.features]
        y = self.df[self.target]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.2, random_state=42)

    def build(self):
        clf = DecisionTreeClassifier()
        self.model = clf.fit(self.x_train,self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.x_test)

    def evaluate(self):
        dict = {"Actual":self.y_test,"Predicted":self.y_pred}
        print(dict)
        print("Accuracy:", metrics.accuracy_score(self.y_test, self.y_pred))