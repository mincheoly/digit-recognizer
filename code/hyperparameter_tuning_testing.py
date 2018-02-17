import csv
import collections
import random
import numpy as np
import math
import string
from util import *
from testing import *
from lc_hyperparameter import *

trainingSet = readFile('train.csv', "train")
testSet = readFile('test.csv', "test")

def run_lc(numiters, eta):
	LC = linear_classifier(numiters, eta)
	LC.train(trainingSet)
	return LC.test(testSet)

numiters = 26
eta = 0.5
numiters_jump = 2
eta_jump = 0.1
prev_accuracy = 0.0
curr_accuracy = run_lc(numiters, eta)
i = 1

while (True):
	if eta - i*eta_jump > 0:
		temp_accuracy1 = run_lc(numiters + i*numiters_jump, eta - i*eta_jump)
	else:
		temp_accuracy1 = run_lc(numiters + i*numiters_jump, eta_jump)
	if numiters - i*numiters_jump > 0 and eta - i * eta_jump > 0:
		temp_accuracy2 = run_lc(numiters - i*numiters_jump, eta - i*eta_jump)
	else:
		temp_accuracy2 = run_lc(numiters_jump, eta_jump)
	temp_accuracy3 = run_lc(numiters + i*numiters_jump, eta + i*eta_jump)
	if numiters - i*numiters_jump > 0:
		temp_accuracy4 = run_lc(numiters - i*numiters_jump, eta + i*eta_jump)
	else:
		temp_accuracy4 = run_lc(numiters_jump, eta + i*eta_jump)
	templist = [temp_accuracy1, temp_accuracy2, temp_accuracy3, temp_accuracy4]
	max_temp_index = templist.index(max(templist))
	if templist[max_temp_index] > curr_accuracy:
		curr_accuracy = templist[max_temp_index]
		if max_temp_index % 2 == 0:
			numiters = max(numiters - i * numiters_jump, 5)
			eta = max(eta + (max_temp_index - 3) * i * eta_jump, 0.1)
		else:
			numiters += i * numiters_jump
			eta += max((max_temp_index - 2) * i * eta_jump, 0.1)
		i = 1
	else:
		i += 1
	print "accuracy, numiters, eta, i: " + str(curr_accuracy) + ", " + str(numiters) + ", " + str(eta) + ", " + str(i)