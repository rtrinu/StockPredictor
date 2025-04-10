from src.stock_model.stock import Stock
import os
from flask import Flask, render_template, send_file
from backend.stock_plot_png import plot_close_data
import matplotlib
matplotlib.use('Agg')

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
        filename, filepath = plot_close_data()
        return send_file(filepath, mimetype='image/png')
    except Exception as e:
        print(f"Error generating stock chart: {e}")
        return send_file('path/to/error/image.png', mimetype='image/png'), 500
@app.route('/stock-input')
def stock_input():
    return render_template('stockInput.html')
def main():
    #hybrid = CnnLSTMHybrid.create()
    #randomForest = RandomForestModel.create()
    user_input = input("Input a Stock: ")
    user_stock = Stock.create(user_input)

if __name__ == "__main__":
    app.run()