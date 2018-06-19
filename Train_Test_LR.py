#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from skimage.feature import daisy

print "generating x and y from text"
#(50000, 4096)
unfolded_data = np.genfromtxt('train_x.csv', delimiter=",", dtype='uint8')
x = unfolded_data.reshape((50000,64,64))
#(50000, )
y = np.genfromtxt("train_y.csv", delimiter=",", dtype= 'uint8')

x_test = np.genfromtxt('test_x.csv', delimiter=",", dtype='uint8')
x_test = x_test.reshape((10000,64,64))

print "Applying daisy features to training set"
daisy_features_train_set = np.zeros((len(x),104))
for i in range(len(x)):
    descs, descs_img = daisy(x[i], step=180, radius=20, rings=2, histograms=6, orientations=8, visualize=True)
    daisy_features_train_set[i] = descs.reshape((1,104))

print "Daisy: Saving features' loop for testing set"
daisy_features_test_set = np.zeros((len(x_test),104))
for i in range(len(x_test)):
    descs, descs_img = daisy(x_test[i], step=180, radius=20, rings=2, histograms=6,
                         orientations=8, visualize=True)
    daisy_features_test_set[i] = descs.reshape((1,104))
    
x_train = daisy_features_train_set
y_train = y
x_test = daisy_features_test_set

"""
Logistic Regression
"""
#solver='lbfgs' used for faster memory processing.
logReg = LogisticRegression(C=1e5, solver='lbfgs')
print "Training Logistic Regression"
logReg.fit(x_train,y_train)
print "Testing Logistic Regression"
pred_y = logReg.predict(x_test)
np.savetxt("pred_y.csv",pred_y, delimiter =",")
