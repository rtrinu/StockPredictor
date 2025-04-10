from flask import Flask, render_template
import os

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app', 'backend', 'templates'))



app = Flask(__name__, template_folder=template_dir)



@app.route('/')
def index():
    return render_template('index.html', 
                           title='My Flask App', 
                           message='Welcome to Flask!')

if __name__ == '__main__':
    print("Template Directory:", template_dir)
    
    # Verify directory exists
    if not os.path.exists(template_dir):
        print(f"Error: Template directory does not exist at {template_dir}")
    app.run()