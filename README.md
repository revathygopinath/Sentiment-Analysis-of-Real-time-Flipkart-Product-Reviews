# Flipkart Sentiment Analysis  
### Deployed Machine Learning Web Application 🚀

A production-ready **Sentiment Analysis web application** built using Flipkart product reviews.  
The app classifies customer reviews as **Positive** or **Negative** and is deployed on **AWS EC2** using **Streamlit**, ensuring **24/7 availability**.

---

## 🔗 Live Demo
👉 **http://18.61.127.104:8501**

---

## 📌 Project Overview
This project demonstrates an **end-to-end Machine Learning pipeline**, including:
- Text preprocessing and feature extraction
- Model training and evaluation
- Web application development
- Cloud deployment with monitoring

The application runs entirely on the server and does **not depend on the local machine**.

---

## 🎯 Project Objective
To build and deploy a sentiment analysis system that:
- Processes real-world e-commerce product reviews
- Performs text preprocessing and sentiment prediction
- Serves predictions through a user-friendly web interface
- Operates continuously using cloud infrastructure

---

## 🧠 Model Details
- **Algorithm:** Logistic Regression  
- **Vectorization:** Bag of Words (CountVectorizer)  
- **Classes:** Positive / Negative  
- **Imbalance Handling:** `class_weight="balanced"`  
- **Accuracy:** ~91.9%  
- **F1-score:** ~92.2%  

### Model Artifacts
- `sentiment_model.pkl`
- `bow_vectorizer.pkl`
- `model_metadata.json`

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Machine Learning:** scikit-learn  
- **NLP:** NLTK  
- **Web Framework:** Streamlit  
- **Cloud Platform:** AWS EC2  
- **Operating System:** Ubuntu 22.04 LTS  
- **Process Management:** systemd  

---

## 📁 Project Structure

```text
Sentiment-Analysis-of-Real-time-Flipkart-Product-Reviews/
│
├── app/
│   └── app.py                 # Streamlit application
│
├── models/
│   ├── sentiment_model.pkl    # Trained ML model
│   ├── bow_vectorizer.pkl     # CountVectorizer
│   └── model_metadata.json   # Model information
│
├── data/                      # Dataset files (if any)
├── notebooks/                 # EDA & experimentation notebooks
├── scripts/                   # Helper scripts
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

---

## 🚀 Deployment (AWS EC2 + Streamlit)

### Deployment Environment
- **EC2 Instance Type:** t3.micro  
- **Region:** ap-south-2  
- **Inbound Ports Enabled:**
  - SSH – 22
  - Streamlit – 8501

---

### Deployment Steps
1. Launched an EC2 instance with Ubuntu 22.04.
2. Configured security groups to allow SSH and Streamlit access.
3. Cloned the GitHub repository onto the EC2 instance.
4. Created and activated a Python virtual environment.
5. Installed dependencies from `requirements.txt`.
6. Configured the Streamlit app to load trained ML models.
7. Deployed the application as a **systemd service** to ensure:
   - Automatic restart on failure
   - Auto-start on server reboot
   - Continuous availability (24/7)

---

## 🧪 Testing and Monitoring

### Testing
- Verified public accessibility using EC2 public IP.
- Tested predictions using positive, negative, and edge-case reviews.
- Restarted the service to validate recovery.
- Rebooted the EC2 instance to confirm auto-start functionality.

### Monitoring
- Service status monitored using:
  ```bash
  sudo systemctl status flipkart-streamlit
Application logs monitored using:

sudo journalctl -u flipkart-streamlit -f
CPU and memory usage monitored using:

top
df -h
The application is configured with automatic restart to ensure high availability.

✅ Production Highlights
End-to-end ML pipeline (training → inference → deployment)

Cloud-hosted and laptop-independent

Auto-restart and reboot-safe deployment

Real-world deployment using AWS and systemd

⚠️ Notes
The model was trained using scikit-learn 1.5.1 and served using scikit-learn 1.3.x.

Version mismatch generates warnings only; inference remains stable.

The application runs fully on the server and does not depend on the local machine.

👩‍💻 Author
Revathy Gopinath

🔗 GitHub: https://github.com/revathygopinath

📌 Future Improvements
Add multi-class sentiment classification

Store predictions in a database

Integrate CI/CD pipeline

Add analytics dashboard