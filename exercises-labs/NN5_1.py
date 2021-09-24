import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# load the dataset

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
T = np.array([[0], [1], [1], [0]])
# define the keras model
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, T, epochs=300)
# evaluate the keras model
_,accuracy = model.evaluate(X, T)
print('Accuracy: %.2f' % (accuracy*100),'%')