import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix

A=pd.read_csv('auto-mpg.data',delim_whitespace=True,header=None)
A=A.to_numpy(columns=[0,1,2,3,4,5,6])
x=A[:,1:7]
x[:,5]=x[:,5]-min(x[:,5])
x[:,5]=x[:,5]/max(x[:,5])

t=A[:,0]

k=235.214583 #l/100km for one mpg
t=k/t

x_train, x_test, t_train, t_test = train_test_split( x, t, 
            test_size=0.2,random_state=42)

net=MLPRegressor(activation='relu',max_iter=1000, verbose=True, 
    alpha=1e-3,hidden_layer_sizes=(6,6), random_state=1)

net.fit(x_train,t_train)
y_test=net.predict(x_test)
model.fit(tr_X, tr_y)
err=np.sum(abs(t_test-y_test)/abs(t_test))/len(t_test)
loss_values = model.estimator.loss_curve_
print('Mean error for test =',np.round(100*err,2),'%')

plt.close()

plt.plot(loss_values)
plt.title('Loss curve')

plt.figure()
plt.stem(t_test,linefmt='g')
plt.plot(y_test,'.r',markersize=12)
plt.ylim((0, max(t_test)))
plt.grid(True)
plt.title('Target(blue) vs. approximated values(red)' )