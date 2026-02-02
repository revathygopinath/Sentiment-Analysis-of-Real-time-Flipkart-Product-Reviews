git# Flipkart Product Review Sentiment Analysis

## 🎯 Project Overview

This project performs sentiment analysis on real-time Flipkart product reviews for the **YONEX MAVIS 350 Nylon Shuttle**. The system classifies customer reviews as positive or negative and provides insights into customer satisfaction and pain points.

### Key Features
- ✅ Binary sentiment classification (Positive/Negative)
- ✅ Multiple text embedding techniques (BoW, TF-IDF)
- ✅ Various ML models comparison
- ✅ Interactive Streamlit web application
- ✅ Real-time sentiment prediction
- ✅ AWS EC2 deployment ready

---

## 📊 Dataset Information

- **Source**: Flipkart Product Reviews
- **Product**: YONEX MAVIS 350 Nylon Shuttle
- **Total Reviews**: 8,518
- **Features**: Reviewer Name, Rating, Review Title, Review Text, Place, Date, Up Votes, Down Votes

---

## 🏗️ Project Structure

```
flipkart_sentiment_analysis/
├── data/
│   └── data.csv                    # Original dataset
├── notebooks/
│   └── 01_EDA_and_Modeling.ipynb   # Complete analysis notebook
├── models/
│   ├── sentiment_model.pkl         # Trained model
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
│   └── preprocess_function.pkl     # Preprocessing function
├── app/
│   └── app.py                      # Streamlit application
├── scripts/
│   ├── train_model.py              # Model training script
│   └── deploy.sh                   # Deployment script
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd flipkart_sentiment_analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 📓 Running the Jupyter Notebook

1. **Start Jupyter**
```bash
jupyter notebook
```

2. **Open the notebook**
   - Navigate to `notebooks/01_EDA_and_Modeling.ipynb`
   - Run all cells sequentially

The notebook covers:
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Extraction (BoW, TF-IDF)
- Model Training (Logistic Regression, Naive Bayes, Random Forest, SVM, XGBoost)
- Model Evaluation and Comparison
- Model Saving

---

## 🌐 Running the Streamlit App

### Local Development

1. **Navigate to app directory**
```bash
cd app
```

2. **Run Streamlit**
```bash
streamlit run app.py
```

3. **Access the application**
   - Open browser: `http://localhost:8501`

### Features
- Text input for custom reviews
- Sample reviews for testing
- Real-time sentiment prediction
- Confidence scores
- Interactive visualizations
- Probability breakdown

---

## 🤖 Model Details

### Data Preprocessing
1. **Text Cleaning**
   - Remove special characters, URLs, emails
   - Convert to lowercase
   - Remove 'READ MORE' tags

2. **Text Normalization**
   - Tokenization
   - Stopwords removal
   - Lemmatization

3. **Feature Extraction**
   - Bag of Words (BoW)
   - TF-IDF (Term Frequency-Inverse Document Frequency)

### Models Trained
1. Logistic Regression
2. Naive Bayes (MultinomialNB)
3. Random Forest
4. Support Vector Machine (SVM)
5. XGBoost

### Best Model
- **Algorithm**: Logistic Regression with TF-IDF
- **F1-Score**: ~0.95
- **Accuracy**: ~95%
- **Evaluation Metric**: F1-Score (weighted)

---

## ☁️ AWS EC2 Deployment

### Prerequisites
- AWS Account
- EC2 instance (t2.micro or higher)
- SSH key pair
- Security group configured (port 8501 open)

### Deployment Steps

1. **Launch EC2 Instance**
   - AMI: Ubuntu Server 22.04 LTS
   - Instance Type: t2.small (minimum)
   - Storage: 20 GB

2. **Connect to EC2**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Update system**
```bash
sudo apt update
sudo apt upgrade -y
```

4. **Install Python and dependencies**
```bash
sudo apt install python3-pip python3-venv -y
```

5. **Clone repository**
```bash
git clone <repository-url>
cd flipkart_sentiment_analysis
```

6. **Setup virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

7. **Configure security group**
   - Add inbound rule: Custom TCP, Port 8501, Source: 0.0.0.0/0

8. **Run application**
```bash
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

9. **Access application**
   - URL: `http://your-ec2-ip:8501`

### Production Deployment (with PM2)

1. **Install Node.js and PM2**
```bash
sudo apt install nodejs npm -y
sudo npm install -g pm2
```

2. **Create startup script**
```bash
echo "streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0" > start.sh
chmod +x start.sh
```

3. **Start with PM2**
```bash
pm2 start start.sh --name sentiment-app
pm2 save
pm2 startup
```

---

## 📈 Model Performance

### Training Results

| Model | Feature | Accuracy | F1-Score |
|-------|---------|----------|----------|
| Logistic Regression | TF-IDF | 0.9514 | 0.9510 |
| SVM | TF-IDF | 0.9487 | 0.9483 |
| Naive Bayes | TF-IDF | 0.9423 | 0.9418 |
| XGBoost | TF-IDF | 0.9398 | 0.9393 |
| Random Forest | TF-IDF | 0.9372 | 0.9367 |

### Key Insights
- TF-IDF features consistently outperform BoW
- Logistic Regression provides best balance of accuracy and speed
- High F1-scores indicate good precision-recall balance
- Model generalizes well on test data

---

## 🔍 Key Findings

### Sentiment Distribution
- **Positive Reviews**: ~85%
- **Negative Reviews**: ~15%

### Common Pain Points (from negative reviews)
1. Product quality concerns
2. Authenticity issues
3. Damaged/defective products
4. Pricing concerns
5. Delivery problems

### Recommendations
1. Enhance quality control measures
2. Address counterfeit product concerns
3. Improve packaging to prevent damage
4. Review pricing strategy
5. Strengthen customer support

---

## 🧪 Testing the Model

### Using Python Script
```python
from app.app import predict_sentiment, load_model

# Load model
model, vectorizer, preprocess_func = load_model()

# Test review
review = "Excellent product! Best quality. Highly recommended!"
prediction, probabilities = predict_sentiment(review, model, vectorizer, preprocess_func)

print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Confidence: {max(probabilities)*100:.2f}%")
```

### Using Streamlit Interface
1. Open the app
2. Enter a review or select a sample
3. Click "Analyze Sentiment"
4. View results with confidence scores

---

## 📝 API Documentation (for Flask deployment)

### Endpoint: `/predict`
- **Method**: POST
- **Content-Type**: application/json

**Request Body**:
```json
{
  "review": "This is an amazing product!"
}
```

**Response**:
```json
{
  "sentiment": "positive",
  "confidence": 98.5,
  "probabilities": {
    "negative": 0.015,
    "positive": 0.985
  }
}
```

---

## 🛠️ Troubleshooting

### Common Issues

1. **NLTK Data Not Found**
```python
import nltk
nltk.download('all')
```

2. **Port 8501 Already in Use**
```bash
streamlit run app.py --server.port 8502
```

3. **Model Files Not Found**
   - Ensure you've run the notebook to train and save models
   - Check `models/` directory for .pkl files

4. **Memory Issues on EC2**
   - Use at least t2.small instance
   - Consider adding swap space

---

## 📚 Dependencies

### Core Libraries
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning
- nltk: Natural language processing
- xgboost: Gradient boosting

### Visualization
- matplotlib: Static plots
- seaborn: Statistical visualizations
- plotly: Interactive charts

### Web Framework
- streamlit: Web application
- flask: REST API (optional)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👥 Authors

- **Data Science Team**
- Email: support@example.com
- GitHub: @yourusername

---

## 🙏 Acknowledgments

- Flipkart for providing the dataset
- NLTK community for NLP tools
- Streamlit team for the web framework
- Scikit-learn contributors

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Email: support@example.com
- Documentation: [Link to docs]

---

## 🔄 Version History

- **v1.0.0** (2024-01-30)
  - Initial release
  - Basic sentiment analysis
  - Streamlit web interface
  - AWS deployment guide

---

## 🎯 Future Enhancements

- [ ] Multi-class sentiment (Positive, Neutral, Negative)
- [ ] Aspect-based sentiment analysis
- [ ] Real-time data scraping
- [ ] Multi-language support
- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Performance monitoring dashboard

---

**Happy Analyzing! 🚀**
