The training set consists of 26344 3x64x64 8bit-RGB images. There is a unique label for each of the examples, between 0 and 39 inclusively.

The test set consists of 6600 images of the same format.

Here's how to load (and view) this data in numpy:

import numpy
trainX = numpy.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = numpy.load('tinyY.npy') 
testX = numpy.load('tinyX_test.npy') # (6600, 3, 64, 64)

# to visualize only
import scipy.misc
scipy.misc.imshow(trainX[0].transpose(2,1,0)) # put RGB channels last

-----------------------------------------

feedforward.py
	- Our own implementation of the feef-forward neural network.
cnn-complete-pictures.py
	- Our most successful program that is able to generate model that results in accuracy of ~92% on kaggle.
cnn-rotated-chars
	- Experimental method which in the end was not used for getting results at kaggle and where the model was
	trained on pictures of individual characters.
Train_Test_LR
	- Logistic regression implementation, used to predict labels for testing data.
Train_Validation_LR
	- Logistic regression implementation, used to predict labels for training data.
letter-extraction.py
	- Code that was used for extracting characters from the pictures and were subsequently used in cnn-rotated-chars.py.
