from sklearn.preprocessing import StandardScaler
import yaml
import pandas as pd
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
def preprocess_data():
    config = load_config()

    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale specified features
    for feature in config["features"]["to_scale"]:
        if feature in X_train.columns:
            X_train[feature] = scaler.fit_transform(X_train[[feature]])
            X_test[feature] = scaler.transform(X_test[[feature]])
    return X_train, X_test, scaler
