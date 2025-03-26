# Fraud Detection - Credit Card Transactions

## Overview

This project implements a fraud detection model for credit card transactions using LightGBM and XGBoost. The challenge addressed in this project is the significant class imbalance, with a large lack of data for the positive class (fraudulent transactions).

## Project Structure

- **main.py**: Runs the entire pipeline (preprocessing, training, and evaluation).
- **data_processing_n_analysis.py**: Prepares the data and contains analysis functions.
- **Train.py**: Trains the LightGBM and XGBoost models.
- **Test.py**: Evaluates the performance of the models.
- **references.txt**: Contains information about the dataset used.

## Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the project, execute the following command:

```bash
python main.py
```

## Results

- **ROC AUC: 98%**
- **Precision: 91%**
- **Recall (positive class): 85%**

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.