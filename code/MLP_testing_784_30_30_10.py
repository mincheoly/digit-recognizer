import csv
import collections
import random
import numpy as np
import math
import string
from util import *
from linear_classifier import *
from multilayered_perceptron import *
from autoencoder import *


# read in the datasets
my_training_data = readFile('train.csv', mode='train')
my_test_data = readFile('test.csv', mode='test')

numIter_list = [20, 30, 40]
batch_size_list = [5, 10, 20]
eta_list = [3.0, 5.0]

print 'BEGIN TESTING WITH 784 - 30 - 30 - 10 MLPs'

for numIter in numIter_list:
	for batch_size in batch_size_list:
		for et in eta_list:
			print '======Testing with numIters={}, mini_batch_size={}, eta={}'.format(numIter, batch_size, et)
			MLP = multilayered_perceptron([784, 30, 30, 10])
			MLP.train(my_training_data, numIters=numIter, mini_batch_size=batch_size, eta=et)
			MLP.test(my_test_data)





























