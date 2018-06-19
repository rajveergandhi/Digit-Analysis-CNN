#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#TO VIEW THE IMAGE
#import numpy   as np 
#import matplotlib.pyplot as plt 
#x = np.loadtxt("train_x.csv", delimiter=",") # load from text 
#y = np.loadtxt("train_y.csv", delimiter=",") 
#x = x.reshape(-1, 64, 64) # reshape 
#y = y.reshape(-1, 1) 
#plt.imshow(np.uint8(x[0]))
#plt.show()

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from skimage.feature import daisy

print "generating x and y from text"
#(50000, 4096)
unfolded_data = np.genfromtxt('train_x.csv', delimiter=",", dtype='uint8')
x = unfolded_data.reshape((50000,64,64))
#(50000, )
y = np.genfromtxt("train_y.csv", delimiter=",", dtype= 'uint8')


print "Applying daisy features"
daisy_features_train_set = np.zeros((len(x),104))
for i in range(len(x)):
    descs, descs_img = daisy(x[i], step=180, radius=20, rings=2, histograms=6, orientations=8, visualize=True)
    daisy_features_train_set[i] = descs.reshape((1,104))

print "splitting training set"
x_train, x_test, y_train, y_test = train_test_split(daisy_features_train_set, y, test_size = 0.2, random_state = 13)

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

print "Logistic Regression Accuracy",(metrics.accuracy_score(y_test, pred_y))
