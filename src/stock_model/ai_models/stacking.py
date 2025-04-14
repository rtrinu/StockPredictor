import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class StackedModel():
    def __init__(self,df):
        self.df = df
        
    @classmethod
    def create(cls,df):
        self = cls(df)
        self.load_data()
        self.build_models()
        self.train()
        self.predict()
        self.evaluate()

    def load_data(self):
        self.df = self.df.dropna()
        x = self.df.drop(columns=['Signal', 'Date']) 
        y = self.df['Signal']
        self.x_train,self.x_test, self.y_train,self.y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    def build_models(self):
        base_models = [('rf', RandomForestClassifier(n_estimators=100)),
                       ('svc',SVC(probability=True))]
        meta_model = LogisticRegression()
        stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
        self.model = stacking_model
    
    def train(self):
        self.model.fit(self.x_train,self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.x_test)
        return y_pred
    
    def evaluate(self):
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_test,y_pred)
        print(f"Accuracy of Stacked Model: {accuracy:.2f}")
