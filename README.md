# House Rent Prediction

This project predicts house rent using multiple features such as house size, number of bedrooms (BHK), furnishing status, and floor information.

The purpose of this project is to learn how a complete machine learning pipeline works, from data preprocessing to model evaluation.

---

## What the Project Includes

- Cleaning and preparing the dataset
- Converting categorical features using one-hot encoding
- Scaling numerical features using Z-score normalization
- Splitting data into training, validation, and test sets
- Training polynomial regression models (degrees 1 to 3)
- Applying Ridge regularization to reduce overfitting
- Selecting the best model using validation MSE
- Evaluating final performance on a test dataset
- Making predictions on new house inputs

---

## Model Overview

- Regression type: Polynomial Linear Regression
- Regularization: Ridge (L2)
- Evaluation metric: Mean Squared Error (MSE)
- Implementation: scikit-learn

The model with the lowest validation error is selected and then tested on unseen data.

---

## How to Run the Project

```bash
pip install numpy pandas scikit-learn matplotlib
python house_rent_prediction.py
