from abc import ABC, abstractmethod

class AIModel(ABC):
    def __init__(self, df, stock_name:str):
        self.stock_name = stock_name
        self.df = df
        self.model = None

    @classmethod
    @abstractmethod
    def create(cls, df, stock_name:str):
        pass

    @abstractmethod
    def load_and_preprocess_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def prediction(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def run(self):
        pass
        