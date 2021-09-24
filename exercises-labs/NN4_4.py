import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

x=np.linspace(0,2*np.pi,8)
y=np.sin(x)

plt.close()
plt.plot(x,y,'ro')

model = LinearRegression() 
model.fit(x.reshape(-1, 1),y)

# y=ax+b
#a=model.coef_
#b=model.intercept_

yL = model.coef_ * x +  model.intercept_
plt.plot(x, yL)

x=x.reshape(-1, 1)
t=y.reshape(-1, 1)

net = MLPRegressor(solver='lbfgs',activation='logistic', alpha=1e-5, verbose=1, 
                    hidden_layer_sizes=(3), random_state=1)
net.fit(x,t)

xx=np.arange(0,2*np.pi,.05).reshape(-1, 1)
yP=net.predict(xx)  
plt.plot(xx, yP, '.-m')


