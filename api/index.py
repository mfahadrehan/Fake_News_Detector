import os
import sys
import string
import joblib
import numpy as np
import nltk
from flask import Flask, request, jsonify, render_template
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── NLTK data path: Vercel's writable tmp dir ────────────────────────────────
NLTK_DATA_DIR = "/tmp/nltk_data"
nltk.data.path.insert(0, NLTK_DATA_DIR)

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)

# ── Flask app ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "templates")
MODEL_DIR = os.path.join(BASE_DIR, "..", "saved_models")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ── Load model & vectorizer ───────────────────────────────────────────────────
MODEL_PATH      = os.path.join(MODEL_DIR, "fake_news_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model files not found in {MODEL_DIR}")
    model = vectorizer = None
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = vectorizer = None

# ── NLP helpers ───────────────────────────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text  = text.lower()
    text  = "".join(ch for ch in text if ch not in string.punctuation)
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        data       = request.get_json(force=True)
        input_text = data.get("text", "").strip()

        if not input_text:
            return jsonify({"error": "No text provided."}), 400

        cleaned   = clean_text(input_text)
        if not cleaned:
            return jsonify({"prediction_label": "N/A - Input empty after cleaning"})

        vec        = vectorizer.transform([cleaned])
        prediction = model.predict(vec)
        label      = "Real News" if prediction[0] == 1 else "Fake News"
        return jsonify({"prediction_label": label})

    except KeyError:
        return jsonify({"error": 'Missing "text" key in JSON.'}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed."}), 500
