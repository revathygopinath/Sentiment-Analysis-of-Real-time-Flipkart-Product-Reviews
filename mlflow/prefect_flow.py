from prefect import flow, task
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

@task
def load_data():
    df = pd.read_csv("data/data.csv")
    df = df.dropna(subset=["Review text"])
    df["sentiment"] = df["Ratings"].apply(lambda x: 1 if x >= 4 else 0)
    return df

@task
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["Review text"], df["sentiment"], test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    return f1_score(y_test, preds)

@flow(name="Flipkart Sentiment Training Flow")
def flipkart_training_flow():
    mlflow.set_experiment("Prefect_MLflow_Integration")

    with mlflow.start_run(run_name="Prefect_LogReg_BoW"):
        df = load_data()
        f1 = train_model(df)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("vectorizer", "BoW")
        mlflow.log_metric("f1_score", f1)

        print("✅ Flow finished | F1:", f1)

if __name__ == "__main__":
    flipkart_training_flow()
