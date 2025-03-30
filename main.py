from src.stock_model.ai_models.lstmcnnHybrid import CnnLSTMHybrid
from src.stock_model.ai_models.randomforest import RandomForestModel
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
    #hybrid = CnnLSTMHybrid.create()
    randomForest = RandomForestModel.create()


if __name__ == "__main__":
    main()