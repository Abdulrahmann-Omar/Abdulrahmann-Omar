# Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset used for this project is sourced from Kaggle's "Credit Card Fraud Detection" dataset, which contains anonymized transaction data and a binary label indicating whether a transaction is fraudulent or not.

## Project Overview

The primary goal of this project is to build and evaluate a machine learning model that can accurately classify transactions as fraudulent or legitimate. The dataset is highly imbalanced, with a very small percentage of transactions being fraudulent, making this a challenging classification problem.

## Features

- **Data Preprocessing**: Handling class imbalance, scaling features, and splitting data into training and testing sets.
- **Exploratory Data Analysis (EDA)**: Visualizing patterns in the dataset and understanding the distribution of features.
- **Model Training**: Implementing and training multiple machine learning algorithms such as Logistic Regression, Random Forest, and Gradient Boosting.
- **Evaluation**: Assessing model performance using metrics such as precision, recall, F1-score, and ROC-AUC.
- **Hyperparameter Tuning**: Fine-tuning model parameters to optimize performance.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - XGBoost (if applicable)

## Dataset

The dataset contains transactions made by European cardholders in September 2013. It consists of 31 features:
- **Features V1 to V28**: Result of a PCA transformation.
- **Time**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **Amount**: Transaction amount.
- **Class**: Binary label (1 for fraudulent transactions, 0 for legitimate transactions).

### Dataset Source
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
2.Install the required libraries:

   ```bash
pip install -r requirements.txt
Download the dataset from Kaggle and place it in the data/ directory: Kaggle Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```
3. Run the notebook:
 ```bash
jupyter notebook Fraud_Detection_ML_Final.ipynb
```
## Challenges
Imbalanced Dataset: The dataset contains only 0.17% fraudulent transactions, requiring careful handling of class imbalance.
Feature Engineering: Understanding and working with anonymized PCA features.

## Future Improvements
Incorporate deep learning models such as autoencoders for anomaly detection.
Deploy the model as a web application using Flask or FastAPI.
