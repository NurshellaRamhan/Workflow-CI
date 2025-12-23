import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():
    # 1. Load data hasil preprocessing (WINE)
    train_df = pd.read_csv("wine_preprocessing/wine_train.csv")
    test_df = pd.read_csv("wine_preprocessing/wine_test.csv")

    X_train = train_df.drop("quality", axis=1)
    y_train = train_df["quality"]

    X_test = test_df.drop("quality", axis=1)
    y_test = test_df["quality"]

    # 2. Aktifkan MLflow autolog
    mlflow.set_experiment("Wine Quality Classification")
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        # 3. Train model
        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)

        # 4. Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("Accuracy:", acc)


if __name__ == "__main__":
    train_model()
