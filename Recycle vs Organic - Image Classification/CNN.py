from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sb
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

source = 'Dataset/RecycleImages/'

# STATIC PARAMETERS
inputSize  = 32
noFilters = 32
epochs = 5
batch = 16
filterSize = 5
maxpool = 2
stepsPerEpoch = 20000//batch

test_generator = ImageDataGenerator(rescale = 1./255)

#declaring a new Sequential model 
model = Sequential()

#We can now add the first convolutional layer, with 32 filters
model.add(Conv2D(noFilters, (filterSize, filterSize), input_shape = (inputSize, inputSize, 3), activation = 'relu'))

#max pooling layer
model.add(MaxPooling2D(pool_size = (maxpool, maxpool)))
model.add(Conv2D(noFilters, (filterSize, filterSize), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (maxpool, maxpool)))

#To add a Flatten layer
model.add(Flatten())

#fully connected layer with 128 nodes
model.add(Dense(units = 128, activation = 'relu'))

# Set 50% of the weights to 0 - > reduce overfitting
model.add(Dropout(0.5))

#fully connected layer
model.add(Dense(units = 1, activation = 'sigmoid'))
# 1 node because is binary classif

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
metrics = ['accuracy'])
# train our model in real time

train_generator = ImageDataGenerator(rescale = 1./255)
 
#load train set and test set
train_set = train_generator.flow_from_directory(source+'Train/',shuffle = False, color_mode="rgb",
target_size = (inputSize, inputSize),batch_size = batch,class_mode = 'binary')

test_set = test_generator.flow_from_directory(source+'Test/',shuffle = False, color_mode="rgb",
target_size = (inputSize, inputSize),batch_size = batch,class_mode = 'binary')

model.fit_generator(train_set, steps_per_epoch = stepsPerEpoch, epochs = epochs, verbose=1, validation_data=test_set, validation_steps=100 )

# get general accuracy from the validation of the test set
score = model.evaluate_generator(test_set, steps=100)

for x, metric in enumerate(model.metrics_names):
    print("{} <--> {}".format(metric, score[x]))
    
print('---------------------------------------------------')

probabilities = model.predict_generator(generator=test_set)
print(probabilities)

y_true = test_set.classes
valid_test = model.predict_generator(test_set, steps=20)
metrics.accuracy_score(y_true, probabilities.round(), normalize=True)

plt.close()
print(y_true)
mat = confusion_matrix(y_true, probabilities.round())
print(mat)
ax = sb.heatmap(mat/100, annot=True, 
                 xticklabels=['Organic','Recycle'],
                 yticklabels=['Organic','Recycle'], 
                 cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")



