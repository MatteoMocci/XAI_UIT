import os
import numpy as np
import pandas as pd
from omnixai.data.timeseries import Timeseries
from omnixai.explainers.timeseries import TimeseriesExplainer
from omnixai.visualization.dashboard import Dashboard
import time
from sklearn.svm import OneClassSVM
import joblib
from statsmodels.tsa.seasonal import seasonal_decompose



def train_svm(train_df, model_file):
    model = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
    model.fit(train_df["values"].values.reshape(-1, 1))
    joblib.dump(model, model_file)

def load_svm(model_file):
    model = joblib.load(model_file)
    return model


def timeseries_explain_script(data_file=None, socket=None, model_name=None):
    if model_name is None:
        model_name = "Seasonal Decompose"
    if data_file is not None:
        df = pd.read_csv(data_file)
    else:
        df = pd.read_csv("timeseries.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    df = df.rename(columns={"horizontal": "values"})
    df = df.set_index("timestamp")
    df = df.drop(columns=["anomaly"])

    # Split the dataset into training and test splits
    train_df = df.iloc[:450]
    test_df = df.iloc[450:500]

    if model_name == 'Seasonal Decompose':
        # Seasonal decomposition
        result = seasonal_decompose(train_df["values"], model="additive", period=7)
        
        # The detector function using seasonal component
        def detector(ts: Timeseries):
            seasonal_component = result.seasonal.values
            # Add your anomaly detection logic here
            # For example, calculate the anomaly score based on the difference between the observed seasonal component and the expected seasonal component
            anomaly_scores = np.abs(ts.values - seasonal_component)
            return np.mean(anomaly_scores)

    elif model_name == 'SVM':
        # Check if model files exist
        svm_model_file = 'svm_model.joblib'
        if os.path.exists(svm_model_file):
            model = load_svm(svm_model_file)
        else:
            # Train SVM model
            train_svm(train_df, svm_model_file)
            model = load_svm(svm_model_file)


        # The detector function using One-Class SVM
        def detector(ts: Timeseries):
            scores = model.decision_function(ts.values.reshape(-1, 1))
            return np.mean(scores)

    else:
        raise ValueError("Model not supported!")

    # Initialize a TimeseriesExplainer
    explainers = TimeseriesExplainer(
        explainers=["shap","ce", "mace"],
        mode="anomaly_detection",
        data=Timeseries.from_pd(train_df),
        model=detector,
        preprocess=None,
        postprocess=None,
        params={"mace": {"threshold": 0.001}, "ce":{"threshold":0.001}}
    )
    # Generate explanations
    test_instances = Timeseries.from_pd(test_df)
    local_explanations = explainers.explain(
        test_instances,
        params={"shap": {"nsamples": 400}}
    )

    dashboard = Dashboard(instances=test_instances, local_explanations=local_explanations)

    if socket is not None:
        time.sleep(5)
        socket.emit('dashboard_status', {'running': True})

    dashboard.show()

if __name__ == '__main__':
    timeseries_explain_script()