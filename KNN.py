#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:09:11 2024

@author: cornelius
"""

# ---------------------------- IMPORTS ---------------------------------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



# ------------------------- DATA PREPARATION --------------------------

df = pd.read_csv("gender_classification.csv")
pd.set_option('display.max_columns', None)
print(df)

X = df.drop(columns=['gender'])
y = df['gender']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# ------------------------  K Nearest Neighbors ------------------------

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("The KNN Accuracy is: ", accuracy)