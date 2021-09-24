from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

import warnings  # :D Warnings off
warnings.filterwarnings("ignore")


work=np.array([20, 15, 10 ,3])
GPA= np.array([10, 9, 6, 2])
# how much to work for a five

work=work.reshape(-1, 1)
GPA=GPA.reshape(-1, 1)

model = LinearRegression() 
model.fit(GPA,work)

b=model.intercept_
a=model.coef_

plt.plot(GPA,work,'o')
plt.xlabel('GPA')
plt.ylabel('work')

x=[GPA[0],GPA[-1]]
y=a*x+b
plt.plot(x,y)
plt.grid(True)


net = MLPRegressor(solver='lbfgs',activation='tanh', alpha=1e-5,
                   hidden_layer_sizes=(3,3), random_state=1)
net.fit(GPA,work)
GPAFiner=np.arange(0,10.1,.1).reshape(-1, 1)
wPredict=net.predict(GPAFiner)  
plt.plot(GPAFiner, wPredict, '-m')

gpa=np.array(5)
wrk=net.predict(gpa.reshape(1, -1))
plt.plot(gpa,wrk,'dr')
print('Work for GPA=5 is', np.round(wrk[0],decimals=2),'hours')
