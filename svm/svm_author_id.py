#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm




### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = svm.SVC(kernel='rbf', gamma='auto', C=10000.)
clf.fit(features_train, labels_train)
#t0 = time()
predictions = clf.predict(features_test)
#print clf.score(features_test, labels_test)
#print "training time:", round(time()-t0, 3), "s"
count = 0
for prediction in predictions:
    if prediction == 1:
        count += 1
print count

#########################################################
### your code goes here ###

#########################################################


