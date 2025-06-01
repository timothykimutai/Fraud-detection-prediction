import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data.make_dataset import make_dataset
from src.models.train_model import train_model
from src.models.evaluate import evaluate_model
import joblib
import yaml

def main():
    # Load config
    config_path = project_root / "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Data processing pipeline
    print("Processing data...")
    X_train, X_test, y_train, y_test = make_dataset()
    
    # Model training
    print("Training models...")
    for model_name in ["RandomForest", "LogisticRegression"]:
        train_model(X_train, y_train, model_name)
    
    # Evaluation
    print("Evaluating models...")
    for model_name in ["RandomForest", "LogisticRegression"]:
        # Convert model name to match saved file format
        model_filename = model_name.lower().replace(" ", "").replace("_", "")
        model_path = Path(config['model']['save_dir']) / f"{model_filename}.pkl"
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        evaluate_model(model, X_test, y_test)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()