# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:02:45 2020

@author: Dominic
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt 
import seaborn as sb
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_curve

#read csv
df = pd.read_csv('diabetes.csv')
print(df.head())

plt.close()
# visualize the difference in distribution between diabetics and non-diabetics
plt.figure(1)
for attribute, col in enumerate(df.columns):
    attCoef = plt.subplot(3,3,attribute+1)
    attCoef.yaxis.set_ticklabels([])
# where is not diabetic Outcome == 0 
    sb.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel= False, 
    kde_kws={'linestyle':'-',  
    'color':'blue', 'label':"No Diabetes"})
# where is diabetic Outcome == 1 
    sb.distplot(df.loc[df.Outcome == 1][col], hist=False, axlabel= False, 
    kde_kws={'linestyle':'--', 
    'color':'red', 'label':"Diabetes"})
    attCoef.set_title(col)

# Hide the 9th subplot (bottom right) since there are only 8 plots
#plt.subplot(3,3,9).set_visible(False)

print(df.describe())
print(df.isnull().any())

print("Number of rows with 0 values for each variable")
for col in df.columns:
    missing_rows = df.loc[df[col]==0].shape[0]
    print(col + ": " + str(missing_rows))
print("-----------------------")
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

print("Number of rows with 0 values for each variable")
for col in df.columns:
    missing_rows = df.loc[df[col]==0].shape[0]
    print(col + ": " + str(missing_rows))
print("-----------------------")

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

print("Solved values")
for col in df.columns:
    missing_rows = df.loc[df[col]==0].shape[0]
    print(col + ": " + str(missing_rows))
print("-----------------------")

# standardization is to transform the numeric variables so that each variable has zero mean and unit variance.
df_scaled = preprocessing.scale(df)
# convert to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled['Outcome'] = df['Outcome']
df = df_scaled

#mean, standard deviation and max 
print(df.describe().loc[['mean','std','max'],].round(2).abs())

# SPLITTING TO TEST-25% TRAIN-75% AND VALIDATION SETS
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SPLIT TO VALID 20% and TRAIN 80%
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
#MLS USAGE

model = Sequential()
#input layers
model.add(Dense(32, activation='relu', input_dim=8))
model.add(Dense(16, activation='relu'))

#output layer
model.add(Dense(1, activation='sigmoid'))

#compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#train model for X epochs
model.fit(X_train, y_train, epochs=200)

# Accuracy rises as the no of epochs increases
#evaluate model after loss and accuracy

scores = model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

plt.figure(2)
#confusion matrix
y_test_pred = model.predict_classes(X_test)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sb.heatmap(c_matrix, annot=True, 
                 xticklabels=['No Diabetes','Diabetes'],
                 yticklabels=['No Diabetes','Diabetes'], 
                 cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")

#ROC plotting // positive true test graph 
plt.figure(3)
y_test_pred_probs = model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


plt.show()




