import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix

X=pd.read_excel('auto-mpg.csv', header=None)

x=X.values

t=x[:,0]
x=x[:,1:6]

x=x.astype(float)
t=t.astype(float)
k=235.214583
t=k/t # l/100km

x_train, x_test, t_train, t_test = train_test_split( x, t, 
            test_size=0.2,random_state=42)

net=MLPRegressor(activation='relu',max_iter=5000, verbose=True, 
    alpha=1e-3,hidden_layer_sizes=(3,5), random_state=10)

net.fit(x_train,t_train)
y_test=net.predict(x_test)

err=np.sum(abs(t_test-y_test)/abs(t_test))/len(t_test)
print('Mean error for test =',np.round(100*err,2),'%')

plt.close()
plt.plot(net.loss_curve_)


plt.figure()
plt.plot(t_test,'og')
plt.plot(y_test,'dr')
