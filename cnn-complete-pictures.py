import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras import backend as K
from keras.constraints import maxnorm

# Load training data
X_train = []
with open('train_x_small.csv','r') as f:
    for i,l in enumerate(f):
        X_train.append([int(float(a)) for a in l.split(',')])
X_train = np.array(X_train)
X_train = X_train.reshape(-1,64,64, 1)


# Load training labels
Y_letter_true = []
with open('train_y.csv','r') as f:
    for l in f:
        Y_letter_true.append(l[:-1])


num_classes = len(set(Y_letter_true))
mapping = {n:i for i,n in enumerate(list(set(Y_letter_true)))}

# Convert labels into one-hot vectors
Y_train = keras.utils.to_categorical([mapping[n] for n in Y_letter_true], num_classes)

# Split training set into development and validation sets
X_train_train, X_train_test, Y_train_train, Y_train_test = train_test_split(X_train, Y_train)

# Convert pixel values from 0-255 range to 0-1 range
X_train_train = X_train_train.astype('float32')
X_train_test = X_train_test.astype('float32')
X_train_train /= 255.0
X_train_test /= 255.0
input_shape = (64, 64, 1)

# Create trinarized images by applying two thresholds to the image color values
binarizer = lambda n: 1.0 if n > 0.92 else (0.0 if n < 0.8 else 0.0)
bf = np.vectorize(binarizer)

X_train_train = bf(X_train_train)
X_train_test = bf(X_train_test)

# CNN setup

batch_size = 16
epochs = 25

model = Sequential()
# Pass 3x3 window to the neurons
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))

# Layers that were added/removed duing testing phase
model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
#model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
#model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
#model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))

model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train_train, Y_train_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_train_test, Y_train_test))
score = model.evaluate(X_train_test, Y_train_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Load testing data
X_test = []
with open('test_x_small.csv','r') as f:
    for i,l in enumerate(f):
        X_test.append([int(float(a)) for a in l.split(',')])
X_test = np.array(X_test)
X_test = X_test.reshape(-1,64,64, 1)


# In[21]:

# Convert image colors from 0-255 range to 0-1 range
X_test = X_test.astype('float32')
X_test /= 255.0

# Apply the picture trinarization function to the testing set
X_test = bf(X_test)

# Predict labels for the testing set and save the output into a file that conforms to the kaggle format
predictions = []

for i, _ in enumerate(X_test):
    prediction = model.predict_classes(X_test[i:(i+1)], verbose=0)[0]
    predictions.append(list(mapping.keys())[list(mapping.values()).index(prediction)])

with open("test_y.csv", "w") as file:
    file.write("Id,Label\n")
    for i in range(0, len(predictions)):
        file.write(str(i+1)+","+str(predictions[i]) + "\n")