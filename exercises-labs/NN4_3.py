import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

x=np.array([0,1,2,3,4,5,6,7,8,9])
y=np.array([-1.26, 1.11, 0.81, 2.57, 4.51, 3.24, 5.5, 5.07, 8.09, 8.16])


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

net = MLPRegressor(solver='lbfgs', alpha=1e-5, verbose=1, 
                    hidden_layer_sizes=(3, 3), random_state=1)
net.fit(x,t)
yP=net.predict(x)
plt.plot(x, yP, 'm-d')


