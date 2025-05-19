"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ARTIFACT_DIR = "artifacts"
DATA_PATH = "data/Churn_Modelling.csv"


def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = pd.DataFrame(col_transf.fit_transform(X_train), columns=col_transf.get_feature_names_out())
    X_test = pd.DataFrame(col_transf.transform(X_test), columns=col_transf.get_feature_names_out())
    return col_transf, X_train, X_test, y_train, y_test


def train_model(model_name, X_train, y_train):
    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_name == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    else:
        raise ValueError("Invalid model name")
    model.fit(X_train, y_train)
    return model

    return log_reg

def log_experiment(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        mlflow.set_tag("developer", "Toka")
        mlflow.log_param("model_type", model_name)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_ if hasattr(model, 'classes_') else [0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        plt.figure()
        disp.plot()
        plt.savefig(f"{ARTIFACT_DIR}/{model_name}_confusion_matrix.png")
        plt.close()
        mlflow.log_artifact(f"{ARTIFACT_DIR}/{model_name}_confusion_matrix.png")

        return mlflow.active_run().info.run_id, metrics["f1_score"]


def register_best_models(run_infos):
    client = MlflowClient()
    sorted_runs = sorted(run_infos, key=lambda x: x[1], reverse=True)
    for i, (run_id, f1) in enumerate(sorted_runs[:2]):
        model_uri = f"runs:/{run_id}/model"
        model_name = "ChurnPredictionModel"
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        stage = "Production" if i == 0 else "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage,
            archive_existing_versions=True
        )
        logger.info(f"Model {run_id} registered to stage {stage}")

def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("mlruns")  # Optional: adjust as needed
    mlflow.set_experiment("Bank Churn Prediction")
    ### Set the experiment name


    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run():
        df = pd.read_csv("data/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the max_iter parameter
        mlflow.log_param("max_iter", 1000)
        run_infos = []
    for model_name in ["logistic", "random_forest", "xgboost"]:
        model = train_model(model_name, X_train, y_train)
        run_id, f1 = log_experiment(model_name, model, X_train, X_test, y_train, y_test)
        run_infos.append((run_id, f1))

    register_best_models(run_infos)


if __name__ == "__main__":
    main()