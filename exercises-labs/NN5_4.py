import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load full data data from https://www.openml.org/d/554
X, T = fetch_openml('mnist_784', version=1, return_X_y=True)

# test to see the images and labels are OK
index=0
im=X[index]
imSize=int(len(im)**.5)
im=np.reshape(im,[imSize,imSize])
plt.imshow(im,cmap=plt.cm.gray_r)
plt.title('Digit '+T[index])

X = X / 255. # normalization

# no shuffle needed
xTrain, xTest = X[:60000], X[60000:]
tTrain, tTest = T[:60000], T[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

mlp.fit(xTrain, tTrain)
yTest=mlp.predict(xTest)
print("Training set score: %f" % mlp.score(xTrain, tTrain))
print("Test set score: %f" % mlp.score(xTest, tTest))

print('The accuracy is:',accuracy_score(tTest,yTest)) # accuracy_score(y_true, y_pred)
print('Confusion Matrix is: ') 
print(confusion_matrix(tTest,yTest)) # confusion_matrix(y_true, y_pred) - ON LINES!!!



