# Fraud-detection-prediction

A machine learning-based system for detecting fraudulent credit card transactions in real-time.

## Overview

This project implements a real-time credit card fraud detection system using machine learning. It features a web-based dashboard for transaction monitoring and analysis, with the following key capabilities:

- Real-time transaction fraud probability assessment
- Interactive dashboard for transaction monitoring
- Historical transaction analysis
- Configurable fraud detection thresholds
- Support for multiple machine learning models

## Features

- **Real-time Detection**: Instant fraud probability assessment for new transactions
- **Interactive Dashboard**: 
  - Transaction submission and monitoring
  - Historical transaction view
  - Fraud statistics and trends
  - Configurable settings
- **Multiple Models**: Support for both Random Forest and Logistic Regression models
- **Customizable Thresholds**: Adjustable fraud detection sensitivity
- **Transaction History**: Export and analysis of past transactions

## Technical Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Data Processing**: pandas, numpy
- **Model Training**: Random Forest, Logistic Regression with SMOTE for handling class imbalance

## Project Structure

```
credit-card-fraud-detection/
├── app/
│   ├── templates/
│   │   └── dashboard.html
│   └── app.py
├── config/
│   ├── config.yaml
│   └── model_params.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   └── analyze_distributions.py
│   └── models/
│       ├── train_model.py
│       └── evaluate.py
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/timothykimutai/Fraud-detection-prediction
   cd Fraud-detection-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your credit card transaction dataset in `data/raw/creditcard.csv`

5. Train the model:
   ```bash
   python src/models/train_model.py
   ```

6. Run the application:
   ```bash
   python app/app.py
   ```

## Usage

1. Access the dashboard at `http://localhost:5000`
2. Submit transactions for fraud detection
3. Monitor fraud probabilities and statistics
4. Adjust detection thresholds in the settings panel
5. Export transaction history as needed

## Model Details

The system uses a Random Forest classifier with the following key features:

- Balanced class weights to handle imbalanced data
- SMOTE for synthetic minority oversampling
- Feature importance analysis
- Configurable model parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Credit card dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- scikit-learn and imbalanced-learn libraries
- Flask web framework
