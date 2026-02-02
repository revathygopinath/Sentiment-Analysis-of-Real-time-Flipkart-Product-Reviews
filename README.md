🛒 Flipkart Sentiment Analysis – Deployed ML Web App

A machine learning–based Sentiment Analysis web application built using Flipkart product reviews.
The app classifies customer reviews as Positive or Negative and is deployed on AWS EC2 using Streamlit with 24/7 availability.

🔗 Live Demo

👉 http://18.61.127.104:8501

🎯 Project Objective

To build and deploy an end-to-end sentiment analysis system that:

Processes real-world e-commerce product reviews

Performs text preprocessing and sentiment prediction

Serves predictions through a user-friendly web interface

Runs independently of the local machine using cloud deployment

🧠 Model Details

Algorithm: Logistic Regression

Vectorization: Bag of Words (CountVectorizer)

Classes: Positive / Negative

Imbalance Handling: class_weight="balanced"

Accuracy: ~91.9%

F1-score: ~92.2%

Model Artifacts

sentiment_model.pkl

bow_vectorizer.pkl

model_metadata.json

🛠️ Tech Stack

Programming Language: Python

Machine Learning: scikit-learn

NLP: NLTK

Web Framework: Streamlit

Cloud Platform: AWS EC2

OS: Ubuntu 22.04 LTS

Process Management: systemd

📁 Project Structure
Sentiment-Analysis-of-Real-time-Flipkart-Product-Reviews/
│
├── app/
│   └── app.py
├── models/
│   ├── sentiment_model.pkl
│   ├── bow_vectorizer.pkl
│   └── model_metadata.json
├── data/
├── notebooks/
├── scripts/
├── requirements.txt
└── README.md

🚀 Deployment (AWS EC2 + Streamlit)
🔹 Deployment Environment

EC2 Instance Type: t3.micro

Region: ap-south-2

Inbound Ports Enabled:

SSH – 22

Streamlit – 8501

🔹 Deployment Steps

Launched an EC2 instance with Ubuntu 22.04.

Configured security groups to allow SSH and Streamlit access.

Cloned the GitHub repository onto the EC2 instance.

Created and activated a Python virtual environment.

Installed dependencies from requirements.txt.

Configured the Streamlit app to load trained ML models.

Deployed the app as a systemd service to ensure:

Automatic restart on failure

Auto-start on server reboot

Continuous availability (24/7)

🧪 Testing and Monitoring
🔹 Testing

Verified public accessibility using EC2 public IP.

Tested predictions using positive, negative, and edge-case reviews.

Restarted the service to validate recovery.

Rebooted the EC2 instance to confirm auto-start functionality.

🔹 Monitoring

Service health monitored using:

sudo systemctl status flipkart-streamlit


Application logs monitored using:

sudo journalctl -u flipkart-streamlit -f


CPU and memory usage monitored using:

top
df -h


The application is configured with automatic restart to ensure high availability.

⚠️ Notes

The model was trained using scikit-learn 1.5.1 and served using scikit-learn 1.3.x.

Version mismatch produces warnings only; inference remains stable.

The application runs fully on the server and does not depend on the local machine.

✅ Production Highlights

End-to-end ML pipeline (training → inference → deployment)

Cloud-hosted and laptop-independent

Auto-restart and reboot-safe deployment

Real-world deployment using AWS and systemd

👩‍💻 Author

Revathy Gopinath

🔗 GitHub: https://github.com/revathygopinath

📌 Future Improvements

Add multi-class sentiment support

Store predictions in a database

Integrate CI/CD pipeline

