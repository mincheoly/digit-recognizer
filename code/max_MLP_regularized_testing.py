import csv
import collections
import random
import numpy as np
import math
import string
from util import *
from linear_classifier import *
from multilayered_perceptron_improved import *
from autoencoder import *


# read in the datasets
my_training_data = readFile('train.csv', mode='train')
my_test_data = readFile('test.csv', mode='test')

print '100 hidden'
MLP = multilayered_perceptron_regularized([784, 100, 10])
MLP.train(my_training_data, numIters=40, mini_batch_size=10, eta=3.0,lamb=0.6 , testing_data=my_test_data)








































"""
#======MAIN-----------------------
train_data = readFile('train.csv')
test_data = readFile('test.csv')
print 'read all data'

weight_vectors = []# one for each digit
for i in range(10):
    weight_vectors.append(learnPredictor(train_data, i) )
    print 'learned {}'.format(i)

num_correct = 0
for example in test_data:
    feature, label = example
    predicted_digit = classify(weight_vectors, feature)
    if predicted_digit == label:
        num_correct = num_correct + 1

print 'testing error: {}'.format(float(num_correct)/len(test_data) )

#=====BASELINE=======================
# test_data = readFile('test.csv')
# print 'read test data'

# num_correct = 0
# for example in test_data:
#     label = example[1]
#     predicted_digit = random.randint(0, 9)
#     if predicted_digit == label:
#         num_correct += 1

# print 'testing error: {}'.format(float(num_correct)/len(test_data) )

"""