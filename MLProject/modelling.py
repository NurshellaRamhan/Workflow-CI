import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():
    # Load data hasil preprocessing
    train_df = pd.read_csv("wine_preprocessing/wine_train.csv")
    test_df = pd.read_csv("wine_preprocessing/wine_test.csv")

    X_train = train_df.drop("quality", axis=1)
    y_train = train_df["quality"]

    X_test = test_df.drop("quality", axis=1)
    y_test = test_df["quality"]

    # Set experiment
    mlflow.set_experiment("Wine Quality Classification")

    # Aktifkan autolog (INI SUDAH START RUN)
    mlflow.sklearn.autolog()

    # Train model (TANPA start_run)
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)


if __name__ == "__main__":
    train_model()
