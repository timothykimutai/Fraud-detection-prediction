import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.features_to_scale = []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        
        # Create time features
        if "Time" in X.columns:
            X["time_hour"] = X["Time"] % (24*60*60) / (60*60)
            X["time_day"] = X["Time"] //(24*60*60)
            
            # Create interaction features
            if 'Amount' in X.columns and 'V1' in X.columns:
                X["Amount_v1_ratio"] = X["Amount"] / (X['V1'] + 1e-6)
            return X
        
