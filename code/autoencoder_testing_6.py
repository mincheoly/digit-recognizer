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
#his_training_data, his_validation_data, his_test_data = load_data_wrapper()
my_training_data = readFile('train.csv', mode='train')
my_test_data = readFile('test.csv', mode='test')

# test the autoencoder paired with the linear classifier
print 'BEGIN TESTING AUTOENCODER 600 HIDDEN (numIters=30, batch=10, eta=3.0)'
AE = autoencoder([784, 600, 784])
AE.train(my_training_data, numIters=30, mini_batch_size=10, eta=3.0)
encoded_training_data = AE.generate_encoded_dataset(my_training_data)
encoded_testing_data = AE.generate_encoded_dataset(my_test_data)
LC = linear_classifier(num_features=600)
LC.train(encoded_training_data)
LC.test(encoded_testing_data)






























