import streamlit as st
import joblib
import time
import json
import os
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Flipkart Sentiment Analysis",
    page_icon="🛒",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.positive-box {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.negative-box {
    background-color: #f8d7da;
    border-left: 5px solid #dc3545;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =========================
# NLTK setup
# =========================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# =========================
# Text preprocessing
# =========================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"read more", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# =========================
# Load model artifacts
# =========================
@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "bow_vectorizer.pkl"))

    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r") as f:
        metadata = json.load(f)

    return model, vectorizer, metadata

model, vectorizer, metadata = load_model()

# =========================
# Header
# =========================
st.title("🛒 Flipkart Product Review Sentiment Analysis")
st.markdown("AI-powered sentiment classifier for Flipkart product reviews")
st.markdown("---")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 📝 How to Use")
    st.markdown("""
    1. Enter a product review  
    2. Click **Analyze Sentiment**  
    3. View prediction & confidence  
    """)

    st.markdown("---")

    st.markdown("## ⚠️ Known Limitations")
    st.warning("""
    - Sarcasm may confuse the model  
    - Neutral reviews are not supported  
    - At least **3 meaningful words** required  
    """)

    st.markdown("---")

    st.markdown("## 📊 Model Info")
    st.info(f"""
    **Algorithm:** {metadata['algorithm']}  
    **Vectorizer:** {metadata['vectorizer']}  
    **F1-Score:** {metadata['f1_score']:.4f}  
    **Accuracy:** {metadata['accuracy']:.4f}  
    """)

# =========================
# Layout
# =========================
col1, col2 = st.columns([2, 1])

# =========================
# Input & Prediction
# =========================
with col1:
    st.markdown("## 📝 Enter Review")

    review_text = st.text_area(
        "Type or paste your product review below:",
        height=150,
        placeholder="Example: Excellent quality and worth the price."
    )

    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if not review_text or len(review_text.strip()) < 3:
            st.warning("⚠️ Please enter a valid review")
        else:
            with st.spinner("Analyzing sentiment..."):
                cleaned = preprocess_text(review_text)

                if len(cleaned.split()) < 3:
                    st.warning("⚠️ At least 3 meaningful words are required for prediction")
                else:
                    start_time = time.time()

                    vectorized = vectorizer.transform([cleaned])
                    prediction = model.predict(vectorized)[0]
                    probabilities = model.predict_proba(vectorized)[0]

                    inference_time = (time.time() - start_time) * 1000

                    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
                    confidence = max(probabilities) * 100

                    if confidence < 60:
                        st.info("⚠️ Model is uncertain. Adding more details may improve accuracy.")

                    st.markdown("---")
                    st.markdown("## 🎯 Prediction Result")

                    if sentiment == "POSITIVE":
                        st.markdown(
                            f"<div class='positive-box'><h3>😊 POSITIVE</h3>"
                            f"<p>Confidence: {confidence:.2f}%</p></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='negative-box'><h3>😞 NEGATIVE</h3>"
                            f"<p>Confidence: {confidence:.2f}%</p></div>",
                            unsafe_allow_html=True
                        )

                    st.caption(f"⏱ Inference time: {inference_time:.1f} ms")

                    with st.expander("📄 View Details"):
                        st.write("**Original Review:**")
                        st.write(review_text)
                        st.write("**Cleaned Review:**")
                        st.write(cleaned)

# =========================
# Stats
# =========================
with col2:
    st.markdown("## 📈 Model Stats")
    st.metric("F1-Score", f"{metadata['f1_score']:.4f}")
    st.metric("Training Samples", f"{metadata['training_samples']:,}")
    st.metric("Avg Inference Time", "< 2 ms")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;'>"
    "Built with Streamlit & Scikit-learn | Flipkart Sentiment Analysis"
    "</div>",
    unsafe_allow_html=True
)
