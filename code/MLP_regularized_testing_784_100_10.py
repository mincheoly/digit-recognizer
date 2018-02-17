import csv
import collections
import random
import numpy as np
import math
import string
from util import *
from linear_classifier import *
from multilayered_perceptron_improved import *



# read in the datasets
my_training_data = readFile('train.csv', mode='train')
my_test_data = readFile('test.csv', mode='test')

numIter_list = [40]
batch_size_list = [10]
eta_list = [3.0]
lamb_list = [0.4, 0.5, 0.6]

print 'BEGIN TESTING WITH 784 - 100 - 10 MLPs'

for numIter in numIter_list:
	for batch_size in batch_size_list:
		for et in eta_list:
			for lam in lamb_list:
				print '======Testing with numIters={}, mini_batch_size={}, eta={}, lamba={}'.format(numIter, batch_size, et, lam)
				MLP = multilayered_perceptron_regularized([784, 100, 10])
				MLP.train(my_training_data, numIters=numIter, mini_batch_size=batch_size, eta=et, lamb=lam)
				MLP.test(my_test_data)





























