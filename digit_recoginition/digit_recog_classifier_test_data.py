# -*- coding: utf-8 -*-
import pylab as pl

from sklearn import svm, metrics, preprocessing

import csv

import time
start_time = time.time()

from numpy import genfromtxt
my_data = genfromtxt('train.csv', delimiter=',')

print time.time() - start_time, "seconds" #took ~41 seconds



start_time = time.time()

images_train = my_data[1:,1:]
images_train = preprocessing.scale(imagestot)
targets_train = my_data[1:,0]

classifier = svm.SVC(kernel = 'poly', C = 100, gamma = 0.001, degree = 3)

# We learn the digits 
classifier.fit(images_train, targets_train)

print time.time() - start_time, "seconds"



my_test_data = genfromtxt('test.csv', delimiter=',')
test = my_test_data[1:,]
test = preprocessing.scale(test)
predicted = classifier.predict(test)

length = len(predicted)
    

with open('pred_test.csv', 'wb') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['ImageId','Label'])
    for y in range(length):
        csv_writer.writerow([y+1,int(predicted[y])])



