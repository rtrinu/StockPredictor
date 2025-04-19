from src.stock_model.stock import Stock
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, send_file, request, session, redirect, jsonify, url_for
from backend.stock_plot_png import plot_close_data
from src.stock_model.stock import Stock
import matplotlib
import secrets
matplotlib.use('Agg')

app = Flask(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
app.secret_key = secrets.token_hex(16)

@app.route('/')
def index():
    return render_template('index.html')

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


@app.route('/get-stock-data',methods=['GET'])
def get_stock_data():
    stock_symbol = request.args.get('stock','').strip().upper()
    if not stock_symbol:
        return "Input a valid symbol", 400
    
    user_stock = Stock.create(stock_symbol)
    user_stock_symbol = user_stock.return_stock_symbol()
    stock_data = user_stock.display_information()
    stock_plot = user_stock.display_plot()
    session['stock_symbol'] = stock_symbol
    
    return render_template('stockDisplay.html',stock_data=stock_data, stock=stock_symbol, chart_filename = 'static.png' )

@app.route('/predict-stock')
def predict_stock():
    stock_symbol = session.get('stock_symbol', '')
    if not stock_symbol:
        return redirect(url_for('stock_input'))
    return render_template('loading.html')
    
@app.route('/start_training')
def start_training():
    stock_symbol = session.get('stock_symbol', '')
    if not stock_symbol:
        return jsonify({'error': 'Stock symbol not set'}), 400

    user_stock = Stock.create(stock_symbol)
    user_stock.create_and_train()
    predictions = user_stock.output_predictions()
    session['predictions'] = predictions
    return jsonify({'success': True})
    
@app.route('/predictions')
def predictions():
    predictions = session.get('predictions', {})
    stock_symbol = session.get('stock_symbol', '')
    return render_template('predictions.html', prediction_data=predictions, stock_symbol=stock_symbol)

def main():
    #hybrid = CnnLSTMHybrid.create()
    #randomForest = RandomForestModel.create()
    user_input = input("Input a Stock: ")
    user_stock = Stock.create(user_input)
    user_stock.create_and_train()
    user_stock.output_predictions()

if __name__ == "__main__":
    #app.run(debug=True)
    main()