import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json

def load_config():
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def analyze_distributions():
    config = load_config()
    
    # Load training data
    X_train = pd.read_csv(Path(config['data']['processed_dir']) / "X_train.csv")
    
    # Analyze V1-V28 distributions
    v_features = [f'V{i}' for i in range(1, 29)]
    distributions = {}
    
    for feature in v_features:
        if feature in X_train.columns:
            distributions[feature] = {
                'mean': float(X_train[feature].mean()),
                'std': float(X_train[feature].std()),
                'min': float(X_train[feature].min()),
                'max': float(X_train[feature].max()),
                'q1': float(X_train[feature].quantile(0.25)),
                'q3': float(X_train[feature].quantile(0.75))
            }
    
    # Save distributions to a JSON file
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "feature_distributions.json"
    with open(output_path, 'w') as f:
        json.dump(distributions, f, indent=4)
    
    print("Feature distributions saved to:", output_path)
    return distributions

if __name__ == "__main__":
    distributions = analyze_distributions()
    print("\nFeature Distributions Summary:")
    for feature, stats in distributions.items():
        print(f"\n{feature}:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  IQR: [{stats['q1']:.3f}, {stats['q3']:.3f}]") 