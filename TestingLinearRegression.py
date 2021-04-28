#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 19:39:57 2021

@author: cyrildineen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import seaborn as sns

df = pd.read_csv("testing.csv")
df1 = df.copy()

#I convert the IDs to strings before converting to ints
allID = np.unique(df1['Loan_ID'].astype(str)) 
idDict = {} 

c = 0

for aid in allID:        
    idDict[aid] = c        
    c = c + 1 
    
#I convert the Property Areas to strings before converting to ints
allProperty = np.unique(df1['Property_Area'].astype(str)) 
propertyDict = {} 

d = 0
for ap in allProperty:        
    propertyDict[ap] = d        
    d = d + 1 
    
#We are replacing the loan id and property areas to ints
df1['Loan_ID'] = df1['Loan_ID'].map(idDict)
df1['Property_Area'] = df1['Property_Area'].map(propertyDict)


#The following lines are replacing instances of strings with int values
df1.loc[df1['Gender']=='Male', 'Gender'] = 2
df1.loc[df1['Gender']=='Female', 'Gender'] = 1

df1.loc[df1['Married']=='Yes', 'Married'] = 2
df1.loc[df1['Married']=='No', 'Married'] = 1

df1.loc[df1['Dependents']== '0' , 'Dependents'] = 1
df1.loc[df1['Dependents']== '1', 'Dependents'] = 2
df1.loc[df1['Dependents']== '2', 'Dependents'] = 3
df1.loc[df1['Dependents']== '3+', 'Dependents'] = 4

df1.loc[df1['Education']=='Graduate', 'Education'] = 2
df1.loc[df1['Education']=='Not Graduate', 'Education'] = 1

df1.loc[df1['Self_Employed']=='Yes', 'Self_Employed'] = 2
df1.loc[df1['Self_Employed']=='No', 'Self_Employed'] = 1

#I drop any rows with null values and then move the remaining rows to a df1_list
df1 = df1.dropna()
df1_list = df1.values.tolist() 


df = df.dropna()
ids = df["Loan_ID"]

#I create a separate list containing just the loanID's
#I remove the rows with null values to match with the prevoius list
ids_list = ids.values.tolist() 

i = 0

#Iterate through each row in the list and run the linear algorithm on each row
#A 0 or 1 will be the result and this is written to new_output
for row in df1_list:
    
    from sklearn.datasets import make_blobs
    # create the inputs and outputs
    X, y = make_blobs(n_samples=290, centers=2, n_features=12, random_state=2)
    # define model
    model = LogisticRegression(solver='lbfgs')
    # fit model
    model.fit(X, y)
    # define input
    new_input = df1_list
    # get prediction for new input
    new_output = model.predict(new_input)
    # summarize input and output
    
#print(new_output)
    
#Iterate through each value in the result output (0 or a 1)
#A 0 means loan will not be approved
#A 1 means loan will be approved

idListLocation = 0
for line in new_output:

    if(line == 0):
        print("The loan will not be approved for ID:", ids_list[idListLocation])
        #print(ids_list[idListLocation])
    else:
        print("The loan will be approved for ID:", ids_list[idListLocation])

    idListLocation += 1

#To ensure each row/ID in the list is accounted for
print(len(new_output))

