import csv
import collections
import random
import numpy as np
import math
import string
from util import *
from testing import *
from lc_falsepos import *

trainingSet = readFile('train.csv', "train")
testSet = readFile('test.csv', "test")

LC = linear_classifier(10, 0.1)
LC.train(trainingSet)
LC.test(testSet)