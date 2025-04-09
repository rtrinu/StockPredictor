from flask import Flask, render_template, send_file, request, make_response, session
from src.App.Backend.stock_plot_png import plot_close_data
import os
app = Flask(__name__)
app.secret_key = os.urandom(24)

if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def index():
    plot_image_path = plot_close_data()
    return render_template('index.html',image_path = plot_image_path)

if __name__ == "__main__":
    app.run()