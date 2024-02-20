import numpy as np
from flask import Flask, render_template, request
import joblib

# Importing the TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the saved model
model = joblib.load('spam_model.joblib')

# Loading the TfidfVectorizer
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Flask app initialization
app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Transform the input text using the loaded vectorizer
        input_data = vectorizer.transform([text])

        # Make a prediction using the loaded model
        prediction = model.predict(input_data)[0]

        return render_template('index.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
