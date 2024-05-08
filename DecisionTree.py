#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:44:17 2024

@author: cornelius
"""

# ---------------------------- IMPORTS ---------------------------------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier



# ------------------------- DATA PREPARATION --------------------------

df = pd.read_csv("gender_classification.csv")
pd.set_option('display.max_columns', None)
print(df)

X = df.drop(columns=['gender'])
y = df['gender']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# ------------------------- DECISION TREE -----------------------------

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

prediction = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("The accuracy of the decision tree is: ", accuracy)


















