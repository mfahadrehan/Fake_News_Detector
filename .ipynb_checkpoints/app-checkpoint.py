import flask
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

app = Flask(__name__)

try:
    nltk.data.find('corpora/stopwords')
    print("NLTK stopwords already downloaded.")
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'fake_news_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or Vectorizer file not found in '{MODEL_DIR}'.")
    print("Please ensure 'fake_news_model.joblib' and 'tfidf_vectorizer.joblib' are present.")
    model = None
    vectorizer = None
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    model = None
    vectorizer = None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or Vectorizer not loaded properly. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
        input_text = data.get('text', None)

        if not input_text:
            return jsonify({'error': 'No text provided in the request.'}), 400

        cleaned_text = clean_text(input_text)

        if not cleaned_text:
            return jsonify({'prediction_label': 'N/A - Input invalid or empty after cleaning'})

        vectorized_text = vectorizer.transform([cleaned_text])


        prediction = model.predict(vectorized_text)

        prediction_label = "Real News" if prediction[0] == 1 else "Fake News"

        return jsonify({'prediction_label': prediction_label})

    except KeyError:
        return jsonify({'error': 'Missing "text" key in JSON request.'}), 400
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)