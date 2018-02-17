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

#this dummy test data for training the XOR function
# dummySet = [
#     ([0,1], [0, 0]),
#     ([1,0], [0, 1]),
#     ([1,0], [1, 0]),
#     ([0,1], [1, 1])
# ]

#this dummy set is from excel, representing 3 pixels
# excel_dummy_training_set = readFile('testing_excel_file.csv', 'train')
# excel_dummy_testing_set = readFile('testing_excel_file.csv', 'test')

# test the multiclass classifier using multiclass hinge loss
LC = linear_classifier()
LC.train(my_training_data)
LC.test(my_test_data)


# test the multilayered perceptron
print '100 hidden'
MLP = multilayered_perceptron_regularized([784, 100, 10])
MLP.train(my_training_data, numIters=40, mini_batch_size=10, eta=3.0, lamb=0.5 , testing_data=my_test_data)





