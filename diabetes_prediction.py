# -*- coding: utf-8 -*-
"""Importing Modules"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

"""Data Collection and Analysis"""

# loading Dataset 
data = pd.read_csv("diabetes.csv")
data.shape

data.head()
data.describe()
data['Outcome'].value_counts(0)

data.groupby('Outcome').mean()

X = data.drop(columns = 'Outcome', axis = 1)
Y = data['Outcome']

"""Data Standarization"""

scaler = StandardScaler()

scaler.fit(X)
standard_data = scaler.transform(X)
X = standard_data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Training Model"""

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

"""Model Evaluation"""

# Accuracy Score
X_train_prediction = classifier.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy Score of Training Data: ",train_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy Score of Testing Data: ",test_data_accuracy)

"""Predcition"""

input_data = (9,119,80,35,0,29,0.263,29) # 10,139,80,0,0,27.1,1.441,57--> 0       9,119,80,35,0,29,0.263,29--> 1
input_data_array = np.asarray(input_data)

input_data_reshape = input_data_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshape)

prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 1:
  print('Diabetic')
else:
  print('Not - Diabetic')