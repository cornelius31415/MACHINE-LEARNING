#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:57:39 2024

@author: cornelius
"""

# ---------------------------- IMPORTS ---------------------------------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# ------------------------- DATA PREPARATION --------------------------

df = pd.read_csv("gender_classification.csv")
pd.set_option('display.max_columns', None)
print(df)

X = df.drop(columns=['gender'])
y = df['gender']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# ------------------------ LOGISTIC REGRESSION -------------------------

log_regression = LogisticRegression()
log_regression.fit(X_train,y_train)

prediction = log_regression.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("The Logistic Regression Accuracy is: ", accuracy)
