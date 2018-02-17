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

print 'BEGIN MAX MLP TESTING'
MLP = multilayered_perceptron([784, 100, 10])
MLP.train(my_training_data, numIters=20, mini_batch_size=10, eta=5.0, testExamples=my_test_data)





























