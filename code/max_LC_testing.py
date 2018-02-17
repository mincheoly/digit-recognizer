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


# test the multiclass classifier using multiclass hinge loss
LC = linear_classifier(numiters=30, Eta=0.1, num_features=784)
LC.train(my_training_data, testExamples=my_test_data)

























