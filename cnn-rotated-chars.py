# Loading extracted numbers and letters from the training samples. Indices in the label CSV
# file correspond to the labels in train_y.csv. Since from each training sample 3 characters
# were extracted, each number/index will be represented 3 times.

import numpy as np
import scipy
import skimage
import sys
import math
from random import randint
from skimage import transform
import keras
from sklearn.utils import shuffle
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.constraints import maxnorm

threshold = 145

# ec = extracted chars
ec_x = []
ec_y = []

# Original assignment training set labels
train_y = []

with open('train_extracted_letters_label.csv','r') as lf:
    for l in lf:
        ec_y.append(l)
            
with open('train_extracted_letters_data.csv','r') as df:
    for l in df:
        img = np.array([int(float(n)) for n in l.split(",")])
        img = img.reshape(32, 32)
        ec_x.append(img)

with open('train_y.csv','r') as f:
    for l in f:
        train_y.append(l)

# Load filtered (i.e. only As and (M/m)s) EMNIST dataset and convert them to binary images.

emnist_x = []
emnist_y = []

cmap = {"A": 10, "M": 11, "m": 12}

with open('filtered_mnist_data.csv','r') as f:
    for l in f:
        img = np.array([int(float(n)) for n in l.split(',')])
        img = img.reshape(28,28)
        emnist_x.append(img.T)

with open('filtered_mnist_ld_labels.csv','r') as f:
    for l in f:
        emnist_y.append(cmap[l.strip()])

org_emnist_x_len = len(emnist_x)

# Load original MNIST dataset.

from keras.datasets import mnist

(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()

org_mnist_x_train_len = len(mnist_x_train)
org_mnist_x_test_len = len(mnist_x_test)

# Merge both MNIST and EMNIST datasets together.
meging_disabled = False

if not meging_disabled and org_emnist_x_len + org_mnist_x_train_len > len(mnist_x_train):
    print(len(mnist_x_train), len(mnist_y_train))
    mnist_x_train = np.concatenate((mnist_x_train, emnist_x))
    mnist_y_train = np.concatenate((mnist_y_train, emnist_y))
    print(len(mnist_x_train), len(mnist_y_train))

# Expand original MNIST with randomly rotated versions of the image

# Number of new artificially generated rotated versions of the images
p_rotated_images = 50
n_rotated_images = int((p_rotated_images / 100) * org_mnist_x_train_len)
new_width, new_height = 28, 28

mnist_x_train_len = len(mnist_x_train)
last_rand_index = 0
last_rand_degree = 0

new_samples = []
new_labels = []

# Perform this step only if we have not yet generated these extra images
if mnist_x_train_len < n_rotated_images + org_mnist_x_train_len:
    for i in range(0, n_rotated_images):
        rand_index = randint(0, mnist_x_train_len - 1)
        
        # Only rotate the images in increments of 5 degrees
        rand_degree = randint(0, 18) * 5 - 45
        last_rand_index = rand_index
        last_rand_degree = rand_degree
        rotated_mnist = scipy.ndimage.interpolation.rotate(mnist_x_train[rand_index], rand_degree)
        
        # After rotation, the image has dimensions greater than 28x28. Crop only the center 28x28
        rho = (len(rotated_mnist)-new_height)//2
        rwo = (len(rotated_mnist[0])-new_width)//2
        new_mnist = rotated_mnist[rho:(rho + new_height), rwo:(rwo + new_width)]

        new_samples.append(new_mnist)
        new_labels.append(mnist_y_train[rand_index])
        
        if (i+1) % 1000 == 0:
            sys.stdout.write("\r%i/%i done" % (i+1, n_rotated_images))
            sys.stdout.flush()

    mnist_x_train = np.concatenate((mnist_x_train, new_samples))
    mnist_y_train = np.concatenate((mnist_y_train, new_labels))

# Convert MNIST data into binary images.

datasets = [mnist_x_train, mnist_x_test]
dslen = len(datasets)
for d, dataset in enumerate(datasets):
    dlen = len(dataset)
    for i, img in enumerate(dataset):
        for j, row in enumerate(img):
            for k, val in enumerate(row):
                if val > threshold:
                    datasets[d][i][j][k] = 1
                else:
                    datasets[d][i][j][k] = 0
            
        if (i+1) % 5000 == 0:
            sys.stdout.write("\r%i/%i in %i/%i done   " % (i+1, dlen, d+1, dslen))
            sys.stdout.flush()

# The same thing for emnist has already been done at the loading time of those data.


# Reshape extracted chars to 28x28 images so that they match the (E)MNIST datasets.

resized_ec_x = []

for i, img in enumerate(ec_x):
    resized_ec_x.append(skimage.transform.resize(ec_x[i], (new_height, new_width), preserve_range=True))

# Transform each pixel value in (E)MNIST data into keras compatible pixel
# value (i.e. place each value in an array) and setup some variables for NN.


batch_size = 64
num_classes = len(set(mnist_y_train))
epochs = 25
img_w, img_h = (28, 28)

resized_ec_x = np.array(resized_ec_x)


# "k" at the beginning of the variable names means "keras"
kmnist_x_train = mnist_x_train.reshape(mnist_x_train.shape[0], img_h, img_w, 1)
kmnist_x_test = mnist_x_test.reshape(mnist_x_test.shape[0], img_h, img_w, 1)
kmnist_y_train = keras.utils.to_categorical(mnist_y_train, num_classes)
kmnist_y_test = keras.utils.to_categorical(mnist_y_test, num_classes)

print(kmnist_y_train[60000])
kmnist_x_train, kmnist_y_train = shuffle(kmnist_x_train, kmnist_y_train)
print(kmnist_y_train[60000])

# Learning the NN.


input_shape = (img_h, img_w, 1)

# input image dimensions
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

model.fit(kmnist_x_train[:90000], kmnist_y_train[:90000],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(kmnist_x_train[90000:], kmnist_y_train[90000:]))


# Make predictions on the actual project's training set

import copy

lmap = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}

def argmax(l):
    return max(enumerate(l), key=lambda i: i[1])[0]

def resolve_prediction(c, n):
    r = -1
    if len(n) == 2:
        if c[0] == 10:
            r = n[0] + n[1]
        if c[0] == 11 or c[0] == 12:
            r = n[0] * n[1]
    return r

prediction_dataset = ec_y[::3][:10000]
result = []
predicted_labels = []
for i, label in enumerate(prediction_dataset):
    c0i = 3 * i + 0
    c1i = 3 * i + 1
    c2i = 3 * i + 2
    
    imgs_to_predict = [resized_ec_x[c0i], resized_ec_x[c1i], resized_ec_x[c2i]]
                    
    predictions = [model.predict(img.reshape(-1, img_h, img_w, 1)) for img in imgs_to_predict]
    
    n = []
    c = []
    
    predicted_label = []
    for p in predictions:
        pn = argmax(p[0].tolist())
        predicted_label.append(pn)
        if pn < 10:
            n.append(pn)
        else:
            c.append(pn)
    
    if len(n) == 3:
        best_n_as_c_index = -1
        best_n_as_c = -1
        best_prob = -1
        for j, p in enumerate(predictions):
            exp = p[0].tolist()[-3:]
            prob = max(exp)
            if prob > best_prob:
                best_prob = prob
                best_n_as_c = argmax(exp) + 10
                best_n_as_c_index = j
                
        del n[best_n_as_c_index]
        c.append(best_n_as_c)
    if len(c) == 2:
        best_c_as_n_index = -1
        best_c_as_n = -1
        best_c = -1
        best_prob = -1
        for j, p in enumerate(predictions):
            if predicted_label[j] < 10:
                continue
            orgexp = p[0].tolist()
            exp1 = orgexp[-3:]
            exp2 = orgexp[:-3]
            prob = -max(exp1)
            if prob > best_prob:
                best_prob = prob
                best_c_as_n = argmax(exp2)
                best_c = predicted_label[j]
        
        c.remove(best_c)
        n.append(best_c_as_n)
    
    r = resolve_prediction(c, n)
    
    predicted_labels.append(predicted_label)
    result.append(r)
    
    if i+1 % 1000 == 0:
        sys.stdout.write("\r%i/%i done" % (i+1, dlen))
        sys.stdout.flush()

# Calculate accuracy for the training set

correct = 0
for i, label in enumerate(prediction_dataset):
    if int(train_y[int(label.strip())]) == result[i]:
        correct += 1

print("Accuracy:", str(correct/len(prediction_dataset)*100) + "%")