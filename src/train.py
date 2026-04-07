import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train():
    df = pd.read_csv("data/churn.csv")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("churn-pipeline")

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5
        )

        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)

        # log metrics
        mlflow.log_metric("accuracy", acc)

        # log params
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)

        # log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Model accuracy: {acc}")

if __name__ == "__main__":
    train()
