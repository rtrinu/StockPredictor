from src.stock_model.stock import Stock
import os
from flask import Flask, render_template, jsonify
from backend.stock_plot_png import plot_close_data

app = Flask(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@app.route('/')
def index():
    return render_template('index.html', 
                           title='My Flask App', 
                           message='Welcome to Flask!')

@app.route('/generate-stock-plot')
def generate_stock_plot():
    try: 
        plot_data = plot_close_data()
        return jsonify(plot_data)
    except Exception as e:
        return jsonify({'error' : str(e)})

def main():
    #hybrid = CnnLSTMHybrid.create()
    #randomForest = RandomForestModel.create()
    user_input = input("Input a Stock: ")
    user_stock = Stock.create(user_input)

if __name__ == "__main__":
    app.run()