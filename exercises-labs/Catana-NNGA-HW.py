#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd# Importing the dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('poker-hand-training-true.data', header=None)

X = dataset.iloc[:, 0:10].values
y = dataset.iloc[:, 10].values

dataset_test = pd.read_csv('poker-hand-testing.data', header=None)
X_test = dataset_test.iloc[:, 0:10].values
y_test = dataset_test.iloc[:, 10].values

sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X, y, batch_size = 10, nb_epoch = 10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.3)

cm = confusion_matrix(y_test, y_pred)