# 📰 Fake News Detector — Vercel Deployment

NLP-based fake news classifier (93% accuracy) — Flask app restructured for Vercel serverless.

## Project Structure

```
fake_news_vercel/
├── api/
│   └── index.py            ← Flask app (Vercel entry point)
├── templates/
│   └── index.html          ← Frontend UI
├── saved_models/
│   ├── fake_news_model.joblib
│   └── tfidf_vectorizer.joblib
├── requirements.txt
├── vercel.json
└── README.md
```

## Deploy to Vercel (Step-by-Step)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit — Vercel deploy"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Deploy on Vercel
1. Go to https://vercel.com and sign in
2. Click **"Add New Project"**
3. Import your GitHub repo
4. Leave all settings as default (Vercel auto-detects `vercel.json`)
5. Click **Deploy** ✅

## Run Locally
```bash
pip install -r requirements.txt
cd api
flask --app index run --port 5000
# Open http://localhost:5000
```

## Tech Stack
- **ML:** Logistic Regression + TF-IDF
- **Backend:** Flask (serverless via Vercel)
- **NLTK:** Stopwords downloaded to `/tmp` on cold start
