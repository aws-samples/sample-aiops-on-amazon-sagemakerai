# Credit Card Fraud Detection Dataset - Column Mapping

This document describes the column names in the `creditcard_predictions_final.csv` dataset.

## Overview

The dataset contains 284,807 credit card transactions with 35 columns total, including 32 feature columns after preprocessing. The column names have been updated to reflect meaningful fraud detection features rather than generic identifiers.

## Column Mapping

| New Column Name | Old Column Name | Description | Type |
|----------------|-----------------|-------------|------|
| `transaction_id` | `Unnamed: 0` | Unique transaction identifier | int64 |
| `transaction_timestamp` | `Time` | Time of transaction | float64 |
| `transaction_hour` | `V1` | Hour of day (0-23) | float64 |
| `transaction_day_of_week` | `V2` | Day of week | float64 |
| `customer_age` | `V3` | Age of cardholder | float64 |
| `account_age_days` | `V4` | Age of account in days | float64 |
| `merchant_category_code` | `V5` | Type of merchant (MCC) | float64 |
| `distance_from_home_km` | `V6` | Distance from home address | float64 |
| `distance_from_last_transaction_km` | `V7` | Distance from previous transaction | float64 |
| `online_transaction` | `V8` | Whether transaction was online | float64 |
| `chip_transaction` | `V9` | Chip card used | float64 |
| `pin_used` | `V10` | PIN entered for transaction | float64 |
| `recurring_transaction` | `V11` | Regular recurring payment | float64 |
| `international_transaction` | `V12` | Cross-border transaction flag | float64 |
| `high_risk_country` | `V13` | Transaction in high-risk country | float64 |
| `num_transactions_24h` | `V14` | Transaction count in last 24 hours | float64 |
| `num_transactions_7days` | `V15` | Transaction count in last 7 days | float64 |
| `avg_transaction_amount_30days` | `V16` | Average amount last 30 days | float64 |
| `max_transaction_amount_30days` | `V17` | Maximum amount last 30 days | float64 |
| `card_present` | `V18` | Physical card present | float64 |
| `address_verification_match` | `V19` | AVS match | float64 |
| `cvv_match` | `V20` | CVV verification match | float64 |
| `velocity_score` | `V21` | Transaction velocity metric | float64 |
| `merchant_reputation_score` | `V22` | Historical merchant score | float64 |
| `time_since_last_transaction_min` | `V23` | Minutes since last transaction | float64 |
| `transaction_type_code` | `V24` | Type of transaction | float64 |
| `customer_tenure_months` | `V25` | Months since account opened | float64 |
| `credit_limit` | `V26` | Card credit limit | float64 |
| `available_credit_ratio` | `V27` | Available credit percentage | float64 |
| `previous_fraud_incidents` | `V28` | Historical fraud on account | float64 |
| `transaction_amount` | `Amount` | Transaction amount in currency | float64 |
| `fraud_prediction` | `prediction` | Model prediction (dropped in training) | bool |
| `fraud_probability` | `probability` | Fraud probability (dropped in training) | float64 |
| `customer_gender` | `gender` | Gender of cardholder | object |
| `is_fraud` | `class` | **Target variable** - Fraud flag | bool |

## Feature Groups

### Transaction Features (5)
Features describing the transaction itself:
- `transaction_timestamp`, `transaction_hour`, `transaction_day_of_week`
- `transaction_amount`, `transaction_type_code`

### Customer Profile (4)
Features about the cardholder:
- `customer_age`, `customer_gender`
- `customer_tenure_months`, `account_age_days`

### Transaction Context (6)
Geographic and temporal context:
- `distance_from_home_km`, `distance_from_last_transaction_km`
- `time_since_last_transaction_min`, `online_transaction`
- `international_transaction`, `high_risk_country`

### Card & Security (5)
Payment security features:
- `chip_transaction`, `pin_used`, `card_present`
- `cvv_match`, `address_verification_match`

### Merchant Information (2)
Merchant-related features:
- `merchant_category_code`, `merchant_reputation_score`

### Behavioral Patterns (7)
Historical transaction behavior:
- `num_transactions_24h`, `num_transactions_7days`
- `avg_transaction_amount_30days`, `max_transaction_amount_30days`
- `velocity_score`, `recurring_transaction`
- `previous_fraud_incidents`

### Credit Information (2)
Credit account details:
- `credit_limit`, `available_credit_ratio`

## Data Preprocessing

### Columns Dropped During Training
These columns are removed before model training:
- `transaction_id` - Not predictive, just an identifier
- `fraud_prediction` - Pre-existing predictions, not used for training
- `fraud_probability` - Pre-existing probability scores, not used for training

### One-Hot Encoding
The `customer_gender` column is one-hot encoded with `drop_first=True`, creating:
- `gender_Male` (1 if Male, 0 otherwise)
- `gender_Other` (1 if Other, 0 otherwise)
- Female is the baseline (encoded as 0 for both)

This results in 32 features total after preprocessing.

## Target Variable

- **Column Name:** `is_fraud`
- **Type:** boolean (converted to int: 0/1)
- **Distribution:** 99.83% non-fraud (0), 0.17% fraud (1)
- **Values:**
  - `0` or `False` = Non-fraudulent transaction
  - `1` or `True` = Fraudulent transaction

## Class Imbalance

The dataset is highly imbalanced with only 492 fraud cases out of 284,807 transactions. This is handled in model training using:
- `scale_pos_weight` parameter in XGBoost (~577.88)
- Stratified train-test split
- Evaluation using ROC-AUC and PR-AUC metrics (not accuracy)

## Usage Notes

1. **Feature Order:** When making predictions, ensure features are in the correct order as they were during training
2. **Preprocessing:** Always apply the same preprocessing (drop columns, one-hot encode gender) before inference
3. **Data Types:** Most features are float64, which is expected for ML model input
4. **Missing Values:** The dataset has no missing values

## Files Updated

The following files have been updated to use the new column names:
- `data/creditcard_predictions_final.csv` - Dataset with renamed columns
- `src/fraud-detection-training.ipynb` - Training notebook
- `src/training/data_preprocessing.py` - Data preprocessing module
- `src/inference/test_endpoint.py` - Endpoint testing module
- `README.md` - Project documentation

## References

This column mapping provides meaningful, interpretable feature names that align with standard fraud detection practices in the financial industry.
