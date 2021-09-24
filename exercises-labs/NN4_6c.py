import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor # MLP Network
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('auto-mpg.data',delimiter=r'\s+', header=None)

X=data.iloc[:,1:7]
T=data[0]

xTrain, xTest, tTrain, tTest = train_test_split(X,T, test_size = 0.2)

net = MLPRegressor(solver='lbfgs', alpha=1e-5, verbose=1, 
                    hidden_layer_sizes=(4, 4), random_state=1)
net.fit(xTrain, tTrain)
yTest = net.predict(xTest)

x=np.arange(len(yTest))

plt.close()
plt.plot(x,tTest,'.-b')
plt.plot(x,yTest,'.-r')