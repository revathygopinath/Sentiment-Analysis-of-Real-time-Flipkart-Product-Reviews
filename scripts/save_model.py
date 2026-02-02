# scripts/save_model.py

import pandas as pd
import re
import os
import json
import joblib
import dill

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =========================
# NLTK setup
# =========================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# =========================
# SELF-CONTAINED preprocessing
# =========================
def preprocess_text(text):
    # Imports INSIDE function (critical for serialization)
    import re
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = str(text).lower()
    text = re.sub(r"read more", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)

# =========================
# Sentiment mapping
# =========================
def create_sentiment(rating):
    if rating <= 2:
        return 0
    elif rating >= 4:
        return 1
    else:
        return 2

# =========================
# Load data
# =========================
print("📥 Loading dataset...")
df = pd.read_csv("data/data.csv")

df["sentiment"] = df["Ratings"].apply(create_sentiment)
df = df[df["sentiment"] != 2].copy()

df["cleaned_review"] = df["Review text"].apply(preprocess_text)
df = df[df["cleaned_review"].str.len() > 0]

X = df["cleaned_review"]
y = df["sentiment"]

print(f"✅ Total samples after cleaning: {len(df)}")

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Vectorization
# =========================
vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# Model training
# =========================
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train_vec, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Evaluation Results:")
print(classification_report(y_test, y_pred))
print(f"Accuracy : {accuracy:.4f}")
print(f"F1-score : {f1:.4f}")

# =========================
# Retrain on full data
# =========================
print("\n🔁 Retraining model on full dataset...")
X_full_vec = vectorizer.fit_transform(X)
model.fit(X_full_vec, y)

# =========================
# Save artifacts
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/bow_vectorizer.pkl")

with open("models/preprocess_pipeline.pkl", "wb") as f:
    dill.dump(preprocess_text, f)

metadata = {
    "model_name": "Flipkart Product Review Sentiment Classifier",
    "algorithm": "Logistic Regression",
    "vectorizer": "CountVectorizer",
    "max_features": 5000,
    "ngram_range": "(1,2)",
    "train_test_split": "80:20",
    "accuracy": round(accuracy, 4),
    "f1_score": round(f1, 4),
    "training_samples": int(len(df)),
    "version": "1.0"
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Model, vectorizer, preprocessing & metadata saved successfully!")
