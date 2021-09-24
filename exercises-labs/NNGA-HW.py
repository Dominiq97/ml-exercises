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

#importez datele de training  aici
dataset = pd.read_csv('adult.data', header=None)

#datele mele de test sunt vreo 65k linii, fiecare linie are 14 coloane, primele 13 sunt input, ultimul e output, aici le separ, pun in x inout și in y output, poți sa descarci setul meu de date sa întelegi, census income se numeste
X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, 14].values

#aici importez datele de test, la fel ca la training
dataset_test = pd.read_csv('adult.data', header=None)
X_test = dataset_test.iloc[:, 0:14].values
y_test = dataset_test.iloc[:, 14].values

#asta e folosit ca sa encode uiesc stringuri, sa ii dau valori numerice
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y);
y_test = labelencoder.fit_transform(y_test);

#aici ii zic la ce coloane sa fac encode, adică alea cu stringuri (exemplu: private, Male, single etc)
for i in [1, 3, 5, 6, 7, 8, 9, 13]:
    labelencoder = LabelEncoder()
    X[:, i] = labelencoder.fit_transform(X[:, i])
    X_test[:, i] = labelencoder.fit_transform(X_test[:, i])
    
#pe asta îl folosesc pentru scale, îmi face valorile mai ok ca sa pot sa le prelucrez
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

#aici inițializez rețeaua neurala
classifier = Sequential()

#astea sunt alea 3 layere, primul e input, al doilea e Hiden, ultimul e output


#aici ii zic ce vreau, adică accuracy și loss, sunt funcții predefinite din keras
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#aici ii zic nr de epoci, cu cât e mai mare numărul de epoci, cu atât o sa ai accuracy mai mare
classifier.fit(X, y, batch_size = 10, nb_epoch = 10)

#asta e pentru prediction, e condiția mea practic, >0.5 in loc de >50k, sa te uiți pe set sa întelegi, e output ul meu practic, ultima coloana de pe fiecare linie
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#asta e confusion matrix, îți zice accuracy dacă calculezi in felul următor: suma de pe diagonala principala supra suma tuturor elementelor ( matricea asta o sa fie mere de 2x2) 
cm = confusion_matrix(y_test, y_pred)