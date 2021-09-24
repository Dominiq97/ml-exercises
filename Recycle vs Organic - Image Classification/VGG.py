from keras.applications.vgg16 import VGG16
from keras.models import Model 
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator 
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import seaborn as sb
import matplotlib.pyplot as plt

source = 'Dataset/RecycleImages/'

# STATIC PARAMETERS
inputSize = 48
batch = 16
stepsPerEpoch = 200
epochs = 3

#The VGG16 model and its trained weights are provided directly in Keras
vgg16 = VGG16(include_top=False, weights='imagenet', 
input_shape=(inputSize,inputSize,3))
#include top=false This argument tells Keras not to import the fully connected layers at the end of the VGG16 network

# Freeze the trained layers - 
for layer in vgg16.layers:
    layer.trainable = True

# Add a fully connected layer with 1 node at the end 
input_ = vgg16.input
output_ = vgg16(input_)
# to flatten the intermediate feature representation before the classifier part(model)
last_layer = Flatten(name='flatten')(output_)
last_layer = Dense(1, activation='sigmoid')(last_layer)
model = Model(input=input_, output=last_layer)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
metrics = ['accuracy'])

trainGenerator = ImageDataGenerator(rescale = 1./255)
testGenerator = ImageDataGenerator(rescale = 1./255)
validGenerator = ImageDataGenerator(rescale = 1./255)

train_set = trainGenerator.flow_from_directory(source+'Train/',
target_size = (inputSize,inputSize),batch_size = batch, class_mode = 'binary')

test_set = testGenerator.flow_from_directory(source+'Test/',shuffle = False, color_mode="rgb",
target_size = (inputSize, inputSize),batch_size = batch,class_mode = 'binary')

# get general accuracy from the validation of the test set
model.fit_generator(train_set, steps_per_epoch = stepsPerEpoch, 
epochs = epochs, verbose=1)

score = model.evaluate_generator(test_set, steps=100)

for x, metric in enumerate(model.metrics_names):
    print("{} <--> {}".format(metric, score[x]))

print('---------------------------------------------------')

probabilities = model.predict_generator(generator=test_set)
print(probabilities)

y_true = test_set.classes
valid_test = model.predict_generator(test_set, steps=20)
metrics.accuracy_score(y_true, probabilities.round(), normalize=False)

plt.close()
print(y_true)
mat = confusion_matrix(y_true, probabilities.round())
print(mat)
ax = sb.heatmap(mat/10, annot=True, 
                 xticklabels=['Organic','Recycle'],
                 yticklabels=['Organic','Recycle'], 
                 cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")