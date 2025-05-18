"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib

### Import MLflow
import mlflow
from mlflow.models.signature import infer_signature

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

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    joblib.dump(col_transf, "column_transformer.joblib") 
    mlflow.log_artifact("column_transformer.joblib")
    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = infer_signature(X_train, log_reg.predict(X_train))
    # Log model
    mlflow.sklearn.log_model(log_reg, "logistic_regression_model", signature=signature)
    ### Log the data
    train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    train_data.to_csv(r"C:\MLOPS\MLOps-Course-Labs\dataset\train_data.csv", index=False)
    mlflow.log_artifact(r"C:\MLOPS\MLOps-Course-Labs\dataset\train_data.csv")
    return log_reg


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    ### Set the experiment name
    mlflow.set_experiment("Churn_Prediction_Experiment")

    df = pd.read_csv(r"C:\MLOPS\MLOps-Course-Labs\dataset\Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    # 1. Logistic Regression
    if mlflow.active_run() is not None:
       mlflow.end_run()

    with mlflow.start_run(run_name="Logistic Regression"):
        mlflow.log_param("max_iter", 1000)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        y_pred = model.predict(X_test)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.set_tag("model", "Logistic Regression")
        plt.figure()
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        ConfusionMatrixDisplay(conf_mat, display_labels=model.classes_).plot()
        plt.savefig("confusion_matrix_lr.png")
        mlflow.log_artifact("confusion_matrix_lr.png")
        plt.close()

    # 2. Random Forest
    if mlflow.active_run() is not None:
       mlflow.end_run()

    with mlflow.start_run(run_name="Random Forest"):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        y_pred = model.predict(X_test)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.set_tag("model", "Random Forest")
        plt.figure()
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        ConfusionMatrixDisplay(conf_mat, display_labels=model.classes_).plot()
        plt.savefig("confusion_matrix_rf.png")
        mlflow.log_artifact("confusion_matrix_rf.png")
        plt.close()

    # 3. Support Vector Classifier
    if mlflow.active_run() is not None:
       mlflow.end_run()

    with mlflow.start_run(run_name="SVC"):
        model = SVC(probability=True)
        model.fit(X_train, y_train)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        y_pred = model.predict(X_test)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.set_tag("model", "SVC")
        plt.figure()
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        ConfusionMatrixDisplay(conf_mat, display_labels=model.classes_).plot()
        plt.savefig("confusion_matrix_svc.png")
        mlflow.log_artifact("confusion_matrix_svc.png")
        plt.close()

if __name__ == "__main__":
    main()
