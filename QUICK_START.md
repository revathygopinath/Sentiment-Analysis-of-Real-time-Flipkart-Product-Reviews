# 🚀 Quick Start Guide - Flipkart Sentiment Analysis

## ⚡ 5-Minute Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- Internet connection

---

## Step 1: Install Dependencies (2 minutes)

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Step 2: Train Model (2 minutes)

### Option A: Run Training Script
```bash
python scripts/train_model.py
```

### Option B: Use Jupyter Notebook
```bash
jupyter notebook
# Open: notebooks/01_EDA_and_Modeling.ipynb
# Run all cells
```

**Expected Output:**
```
✓ Model saved: models/sentiment_model.pkl
✓ Vectorizer saved: models/tfidf_vectorizer.pkl
✓ F1-Score: 0.9510
```

---

## Step 3: Run Web App (1 minute)

```bash
streamlit run app/app.py
```

**Browser automatically opens at:** `http://localhost:8501`

---

## 🎯 Test the Application

### Test Case 1: Positive Review
```
Input: "Excellent product! Original Yonex quality. Fast delivery. Highly recommended!"
Expected: POSITIVE with ~98% confidence
```

### Test Case 2: Negative Review
```
Input: "Very poor quality. Damaged product received. Waste of money. Do not buy!"
Expected: NEGATIVE with ~95% confidence
```

### Test Case 3: Mixed Review
```
Input: "Product quality is good but price is too high for regular use."
Expected: POSITIVE or NEGATIVE (depends on dominant sentiment)
```

---

## 🐛 Common Issues & Quick Fixes

### Issue: NLTK Data Not Found
```bash
python -c "import nltk; nltk.download('all')"
```

### Issue: Port 8501 Already in Use
```bash
streamlit run app/app.py --server.port 8502
```

### Issue: Model Files Not Found
```bash
# Retrain model
python scripts/train_model.py

# Verify files exist
ls -la models/
```

### Issue: Module Not Found
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 📁 Project Structure

```
flipkart_sentiment_analysis/
├── data/
│   └── data.csv                    # Dataset (8,518 reviews)
├── models/
│   ├── sentiment_model.pkl         # Trained model
│   ├── tfidf_vectorizer.pkl        # Feature extractor
│   └── preprocess_function.pkl     # Text preprocessor
├── app/
│   └── app.py                      # Streamlit web app
├── notebooks/
│   └── 01_EDA_and_Modeling.ipynb   # Analysis notebook
├── scripts/
│   ├── train_model.py              # Training script
│   └── deploy.sh                   # Deployment script
├── requirements.txt                # Dependencies
├── README.md                       # Full documentation
├── PROJECT_GUIDE.md                # Detailed guide
└── QUICK_START.md                  # This file
```

---

## 🔧 Command Cheatsheet

### Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Deactivate
deactivate
```

### Application
```bash
# Run Streamlit app
streamlit run app/app.py

# Run on different port
streamlit run app/app.py --server.port 8502

# Run Jupyter notebook
jupyter notebook

# Train model
python scripts/train_model.py
```

### Python Testing
```python
# Quick test in Python
from app.app import load_model, predict_sentiment

model, vectorizer, preprocess = load_model()
text = "Amazing product! Highly recommended!"
prediction, proba = predict_sentiment(text, model, vectorizer, preprocess)

print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Confidence: {max(proba)*100:.2f}%")
```

---

## 📊 Expected Performance

### Training Metrics
- **Training Time**: ~2 minutes
- **Accuracy**: 95.14%
- **F1-Score**: 0.9510
- **Model Size**: ~5 MB

### Inference Metrics
- **Prediction Time**: <100ms
- **Memory Usage**: ~200 MB
- **CPU Usage**: <5%

---

## 🌐 AWS Deployment Quick Start

### 1. Launch EC2 Instance
```bash
# Instance Type: t2.small
# AMI: Ubuntu 22.04
# Storage: 20 GB
# Security Group: Allow ports 22, 8501
```

### 2. Connect and Setup
```bash
ssh -i your-key.pem ubuntu@ec2-ip

# Run deployment script
git clone <your-repo>
cd flipkart_sentiment_analysis
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### 3. Access Application
```
http://your-ec2-public-ip:8501
```

---

## 📚 Next Steps

### For Development
1. ✅ Read `PROJECT_GUIDE.md` for detailed explanations
2. ✅ Explore `notebooks/01_EDA_and_Modeling.ipynb`
3. ✅ Customize `app/app.py` for your needs
4. ✅ Experiment with different models

### For Production
1. ✅ Follow deployment guide in `README.md`
2. ✅ Set up monitoring (PM2, CloudWatch)
3. ✅ Configure HTTPS (Let's Encrypt)
4. ✅ Implement CI/CD pipeline

### For Enhancement
1. ✅ Add more features (aspect-based sentiment)
2. ✅ Implement REST API (Flask/FastAPI)
3. ✅ Try deep learning models (BERT)
4. ✅ Add multi-language support

---

## 💡 Pro Tips

### Tip 1: Faster Training
```python
# Use smaller dataset for quick testing
df_sample = df.sample(n=1000, random_state=42)
```

### Tip 2: Better Predictions
```python
# Ensemble multiple models
predictions = []
for model in [lr_model, nb_model, svm_model]:
    pred = model.predict(X_test)
    predictions.append(pred)

# Majority voting
final_pred = np.round(np.mean(predictions, axis=0))
```

### Tip 3: Cache Everything
```python
# In Streamlit
@st.cache_resource
def load_model():
    return joblib.load('models/sentiment_model.pkl')

@st.cache_data
def preprocess_text(text):
    return clean_text(text)
```

### Tip 4: Monitor Performance
```python
import time

start = time.time()
prediction = model.predict(X)
end = time.time()

print(f"Inference time: {(end-start)*1000:.2f}ms")
```

---

## 🆘 Getting Help

### Documentation
- **Full README**: `README.md`
- **Detailed Guide**: `PROJECT_GUIDE.md`
- **Presentation**: `PRESENTATION.md`

### Support Channels
- 📧 Email: support@example.com
- 🐛 GitHub Issues: [Link]
- 💬 Slack: [Channel Link]
- 📚 Wiki: [Documentation]

### Community Resources
- Stack Overflow: Tag `sentiment-analysis`
- Reddit: r/MachineLearning
- Discord: [ML Community]

---

## ✅ Verification Checklist

Before considering setup complete:

- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list` shows packages)
- [ ] NLTK data downloaded (no import errors)
- [ ] Model trained successfully (files in `models/`)
- [ ] Streamlit app runs without errors
- [ ] Test predictions work correctly
- [ ] Web interface accessible in browser

---

## 🎓 Learning Path

### Beginner (Week 1)
1. Run the quick start guide
2. Understand data preprocessing
3. Test sample predictions
4. Explore Streamlit interface

### Intermediate (Week 2-3)
1. Study model training process
2. Experiment with hyperparameters
3. Add custom features
4. Modify web interface

### Advanced (Week 4+)
1. Implement new models (BERT, RoBERTa)
2. Build REST API
3. Deploy to production
4. Set up monitoring and CI/CD

---

## 🚀 You're Ready!

Congratulations! You now have a fully functional sentiment analysis system.

**What you've achieved:**
✅ Installed and configured environment
✅ Trained ML model with 95%+ accuracy
✅ Deployed interactive web application
✅ Ready to analyze customer reviews

**Time to explore:**
- Analyze your own reviews
- Customize the application
- Deploy to the cloud
- Share with your team

---

## 📞 Questions?

If you encounter any issues not covered here:
1. Check `README.md` for detailed documentation
2. Review `PROJECT_GUIDE.md` for troubleshooting
3. Search existing GitHub issues
4. Create a new issue with details

**Happy Analyzing! 🎉**
