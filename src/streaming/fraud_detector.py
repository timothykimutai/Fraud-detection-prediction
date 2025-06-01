import joblib
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
import numpy as np

class FraudDetector:
    def __init__(self, model_name="RandomForest"):
        try:
            # Get project root and config
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config/config.yaml"
            print(f"Loading config from: {config_path}")
            
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
                
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Convert model name to match saved file format
            model_filename = model_name.lower().replace(" ", "").replace("_", "")
            model_path = project_root / config['model']['save_dir'] / f"{model_filename}.pkl"
            print(f"Looking for model at: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            print(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            print("Model loaded successfully")
            
            # Initialize scaler for Time and Amount
            self.scaler = StandardScaler()
            
        except Exception as e:
            print(f"Error initializing FraudDetector: {str(e)}")
            raise
        
    def preprocess(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """Convert raw transaction dict to feature DataFrame"""
        try:
            # Define the expected feature order
            feature_order = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            
            # Convert to DataFrame and ensure correct feature order
            df = pd.DataFrame([transaction])
            
            # Check for missing features
            missing_features = [f for f in feature_order if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
                
            # Reorder columns to match training data
            df = df[feature_order]
            
            # Convert numeric columns to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Scale Time and Amount
            if 'Time' in df.columns and 'Amount' in df.columns:
                df[['Time', 'Amount']] = self.scaler.fit_transform(df[['Time', 'Amount']])
            
            # Check for any remaining NaN values
            if df.isna().any().any():
                print("Warning: NaN values after preprocessing")
                df = df.fillna(0)  # Replace NaN with 0
            
            # Convert to numpy array to match training data format
            return df.values
            
        except Exception as e:
            print(f"Error in preprocess method: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Traceback:", traceback.format_exc())
            raise
    
    def predict(self, transaction: Dict[str, Any]) -> float:
        """Return fraud probability (0-1)"""
        try:
            # Print input transaction for debugging
            print("Input transaction:", transaction)
            
            features = self.preprocess(transaction)
            print("Preprocessed features shape:", features.shape)
            print("Preprocessed features:", features)
            
            # Check for NaN values
            if np.isnan(features).any():
                print("Warning: NaN values detected in features")
                features = np.nan_to_num(features, nan=0.0)
            
            # Get prediction probabilities
            proba = self.model.predict_proba(features)
            print("Raw prediction probabilities:", proba)
            
            # Check for NaN in probabilities
            if np.isnan(proba).any():
                print("Warning: NaN values in prediction probabilities")
                return 0.5  # Return neutral probability if NaN detected
            
            return float(proba[0, 1])
        except Exception as e:
            print(f"Error in predict method: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Traceback:", traceback.format_exc())
            raise
    