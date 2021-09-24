import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# The digits SMALL sklearn dataset - there are just 1797 samples 
digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# test to see the images and labels are OK
index=39  # the first 39 images are in order, unshuffled
im=data[index]
imSize=int(len(im)**.5)
im=np.reshape(im,[imSize,imSize])
plt.imshow(im,cmap=plt.cm.gray_r)
plt.title('Digit '+str(digits.target[index]))

data=data/data.max()

# Split the data - don't shouffle -it's already done and we want to compare results
xTrain, xTest, tTrain, tTest = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False) 


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



