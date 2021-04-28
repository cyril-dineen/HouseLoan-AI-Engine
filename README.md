# HouseLoan-AI-Engine
The LoanPredictionAlgorithm.py file contains code which opens a CSV file based on properties in Kings county, USA. It replaces values to make it easier to represent on a plot. Train and Test sets are created (80/20 split) 4 models are created and performed on the data

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree
4. K Nearest Neighbour

The Logistic Regression model is the most accurate so our AI engine uses this to calculate the results. 
The TestingRegression Python file returns a 0 or 1 value for each line/loan application in the CSV file to represent whether the applicant should be granted a loan or not.

Reference:
https://www.kaggle.com/imsid777/loan-prediction-project-detailed
