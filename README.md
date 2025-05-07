# 1. Create a virtual environment
python -m venv venv  # For Windows
# or
python3 -m venv venv  # For macOS/Linux

# 2. Activate the virtual environment
.\venv\Scripts\activate  # For Windows
# or
source venv/bin/activate  # For macOS/Linux

# 3. Install dependencies from requirements.txt
pip install -r requirements.txt

# 4. Set up NLTK (Run this script to download necessary NLTK datasets)
python ./setup_nltk.py

# 5. Get your NewsAPI Key:
#       - For submission I have included the API key but included the steps on how to acquire it
#    - Go to https://newsapi.org/ and create an account if you don't have one.
#    - Get your API key from the NewsAPI dashboard.

# 6. Create a .env file and put the API key in there.
#    - Ensure the .env file is located in the root directory and contains the following:
#    ```
#    NEWS_API_KEY=your_api_key_here
#    ```

# 7. Run the Flask app (replace 'app.py' with your main app file name if different)
python ./app.py

# 8. To stop the Flask server, press Ctrl + C

# 9. Deactivate the virtual environment when done
deactivate