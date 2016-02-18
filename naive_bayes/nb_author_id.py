#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
from time import time

from tools.email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "training time: ", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "predicting time: ", round(time()-t0, 3), "s"
print(accuracy_score(labels_test, pred))

# no. of Chris trining emails: 7936
# no. of Sara training emails: 7884
# training time:  1.633 s
# predicting time:  0.089 s
# 0.973265073948


#########################################################


