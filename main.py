from src.stock_model.ai_models.lstmcnnHybrid import CnnLSTMHybrid
from src.stock_model.ai_models.randomforest import RandomForestModel
from src.stock_model.ai_models.exampleforest import method
from src.stock_model.stock import Stock
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
    #hybrid = CnnLSTMHybrid.create()
    #randomForest = RandomForestModel.create()
    user_input = input("Input a Stock: ")
    user_stock = Stock.create(user_input)

if __name__ == "__main__":
    main()