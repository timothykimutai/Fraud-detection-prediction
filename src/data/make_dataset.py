import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from pathlib import Path
import yaml

def load_config():
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def make_dataset():
    config = load_config()

    # Create directories
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(config["data"]["raw_path"])

    # Split data
    X = df.drop(config["features"]["target"], axis=1)
    y = df[config["features"]["target"]]
    
    # Convert to numpy arrays for scikit-learn compatibility
    X = X.values
    y = y.values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"],
        stratify=y
    )

    # Convert back to DataFrame for saving
    X_train_df = pd.DataFrame(X_train, columns=df.drop(config["features"]["target"], axis=1).columns)
    X_test_df = pd.DataFrame(X_test, columns=df.drop(config["features"]["target"], axis=1).columns)
    y_train_df = pd.DataFrame(y_train, columns=[config["features"]["target"]])
    y_test_df = pd.DataFrame(y_test, columns=[config["features"]["target"]])

    # Save processed data
    X_train_df.to_csv(processed_dir / "X_train.csv", index=False)
    X_test_df.to_csv(processed_dir / "X_test.csv", index=False)
    y_train_df.to_csv(processed_dir / "y_train.csv", index=False)
    y_test_df.to_csv(processed_dir / "y_test.csv", index=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    make_dataset()
    
    
