#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:35:49 2020

@author: cyrildineen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = pd.read_csv("/Users/cyrildineen/Desktop/GroupProject/archive/2.csv")

print(df.head)

df.info()

print(df.shape)

print(df.describe(include = 'all'))

#Check for NULL values in the CSV
print(df.isnull().sum())

#Where Loan Amount values are null replace them with the mean Loan Amount value
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())

#Where credit History values are null replace them with the mean Credit History value
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())

#Check if the mean has been put in for Loan Amount and Credit History
print(df.isnull().sum())

#Drop all the other rows in the CSV that have NULL values
df.dropna(inplace=True)

#Check if those NULL value rows have been removed
print(df.isnull().sum())

#Check the shaoe of the CSV file (Rows, Columns)
print(df.shape)

#Compares gender to getting the loan
sns.countplot(df['Gender'],hue=df['Loan_Status'])
print(pd.crosstab(df['Gender'],df['Loan_Status']))

#Compares Married to getting the loan
sns.countplot(df['Married'], hue=df['Loan_Status'])
print(pd.crosstab(df['Married'], df['Loan_Status']))

#Compares Education to getting the loan
sns.countplot(df['Education'],hue=df['Loan_Status'])
print(pd.crosstab(df['Education'],df['Loan_Status']))

#Compares Self_Employmenr to getting the loan
sns.countplot(df['Self_Employed'], hue=df['Loan_Status'])
print(pd.crosstab(df['Self_Employed'],df['Loan_Status']))

#Compares Self_Employmenr to getting the loan
sns.countplot(df["Property_Area"], hue=df['Loan_Status'])
print(pd.crosstab(df['Property_Area'],df['Loan_Status']))

#Replaced Loan_Status from Yes/No to 0 and 1 to make it easier to distinguish
df['Loan_Status'].replace('Y', 1, inplace = True)
df['Loan_Status'].replace('N', 0, inplace = True)

#Count occurences of loan being given/not given (Will be a 0 or 1)
print(df['Loan_Status'].value_counts())

#Replaced Gender from Male/Female to 0 (Female) and 1 (Male) to make it easier to distinguish
df['Gender'].replace('Male', 1, inplace = True)
df['Gender'].replace('Female', 0, inplace = True)

print(df['Gender'].value_counts())

#Replaced Married from Yes/No to 0 and 1 to make it easier to distinguish
df['Married'].replace('Yes', 1, inplace = True)
df['Married'].replace('No', 0, inplace = True)

print(df['Married'].value_counts())

#Replaced Edcuation from Graduare/Not Graduate to 0 and 1 to make it easier to distinguish
df.Education = df.Education.map({"Graduate" : 1, "Not Graduate" : 0})
print(df['Education'].value_counts())

#Replaced Edcuation from Graduare/Not Graduate to 0 and 1 to make it easier to distinguish
df.Dependents = df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
print(df['Dependents'].value_counts())

#Replaced Edcuation from Graduare/Not Graduate to 0 and 1 to make it easier to distinguish
df.Self_Employed = df.Self_Employed.map({'Yes': 1, 'No' : 0})
print(df['Self_Employed'].value_counts())

#Replaced Property_Area from Urban/Rural/Semiurban to 0, 1 and 2 to make it easier to distinguish
df.Property_Area = df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
print(df['Property_Area'].value_counts())

print(df['LoanAmount'].value_counts())

print(df['Loan_Amount_Term'].value_counts())

print(df['Credit_History'].value_counts())

#Showing the correlation matrix
plt.figure(figsize=(16,5))
sns.heatmap(df.corr(),annot=True)
plt.title('Correlation Matrix (for Loan Status)')


print(df.head())
print(df.shape)



#I split the data 80/20
#Train and Test
X = df.iloc[1:542,1:12].values
y = df.iloc[1:542,12].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

print("X_train:", X_train)

print("X_test:", X_test)

print("y_train:", y_train)

print("y_test:", y_test)


#I created a Logistical Regression model (Result is 0.78527) (78.5% accuracy)
model = LogisticRegression()
model.fit(X_train,y_train)

lr_prediction = model.predict(X_test)
print('Logistic Regression accuracy:', metrics.accuracy_score(lr_prediction,y_test))

#I created a SVM model (Support Vector Machine) (Result is 0.6503067) (65.03% accuracy)
model = svm.SVC()
model.fit(X_train,y_train)

svc_prediction = model.predict(X_test)
print("SVC accuracy:", metrics.accuracy_score(svc_prediction, y_test))

#I created a Decision Tree Model (Result is 0.711656) (71.1% accuracy)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

dt_prediction = model.predict(X_test)
print("Decision Tree Accuracy:", metrics.accuracy_score(dt_prediction, y_test))

#I created a K Nearest Neighbours model (Result was 0.61963) 61.9(% accuracy)
model = KNeighborsClassifier()
model.fit(X_train,y_train)

knn_prediction = model.predict(X_test)
print("KNN Accuracy:", metrics.accuracy_score(knn_prediction, y_test))


#Out of all the models I tested the one with the highest % accuracy is Logistic Regression
#Logistic Regression is the AI model to use to predict the desired output
#Credit_Score is the main value/input that the Loan_Approval field relies on the most by looking at the Correlation Matrix