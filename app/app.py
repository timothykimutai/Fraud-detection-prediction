import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from flask import Flask, request, jsonify, render_template
from src.streaming.fraud_detector import FraudDetector
import yaml

app = Flask(__name__)
detector = FraudDetector()

@app.route('/', methods=['GET'])
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction = request.get_json()
        if not transaction:
            return jsonify({"error": "No transaction data provided"}), 400
            
        # Log the received transaction data
        print("Received transaction data:", transaction)
        
        # Validate required fields
        required_fields = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing_fields = [field for field in required_fields if field not in transaction]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
            
        fraud_prob = detector.predict(transaction)
        return jsonify({
            "fraud_probability": fraud_prob,
            "is_fraud": fraud_prob > 0.5
        })
    except Exception as e:
        print("Error processing prediction:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    config_path = project_root / "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("Starting Fraud Detection API...")
    print("Dashboard available at: http://localhost:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)