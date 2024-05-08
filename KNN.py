#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:09:11 2024

@author: cornelius
"""

# ---------------------------- IMPORTS ---------------------------------

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



# ------------------------- DATA PREPARATION --------------------------

df = pd.read_csv("gender_classification.csv")
pd.set_option('display.max_columns', None)
print(df)

label_encoder = LabelEncoder()
df["gender"] = label_encoder.fit_transform(df["gender"])

X = df.drop(columns=['gender'])
y = df['gender']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# ------------------------  K Nearest Neighbors ------------------------

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)


# ------------------------- CONFUSION MATRIX --------------------------

"""
                          Interesting  Metrics
            
                            TP: True Positive
                            TN: True Negative
                            FP: False Positive
                            FN: False Negative

                
            1. Accuracy    =   (TP + TN) / (Total Predictions)
            How often the model is correct
            
            2. Precision   =      TP / (TP + FP)
            What percentage of the positives predicted is truly positive
            
            3. Sensitivity =      TP / (TP + FN)   (also called Recall)
            What percentage of the positives is actually predicted positive
            
            4. F1 Score    =   2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
            Harmonic Mean of Precision and Sensitivity
            Values between 1 and 0. F1 = 1 indicates perfect precision and recall (sensitivity)
                                


"""


conf_matrix = confusion_matrix(y_test, prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, 
                                            display_labels = [0,1])
cm_display.plot()
plt.show()


accuracy = metrics.accuracy_score(y_test, prediction)
precision = metrics.precision_score(y_test, prediction)
sensitivity = metrics.recall_score(y_test, prediction)
f1_score = metrics.f1_score(y_test, prediction)

print("The accuracy of the decision tree is: ", accuracy)
print("The precision of the decision tree is: ", precision)
print("The sensitivity of the decision tree is: ", sensitivity)
print("The F1 Score of the decision tree is: ", f1_score)













