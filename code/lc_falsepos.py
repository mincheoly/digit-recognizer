import csv
import collections
import random
import numpy as np
from util import *

# this linear classifier iterates over constraint values and outputs false pos/false neg
# in addition to accuracy
class linear_classifier:
    def __init__(self, numiters, Eta, num_features=784):
        self.weights = np.array( [[0.0 for i in range(num_features)] for j in range(10)])  # feature => weight
        self.numIters = numiters
        self.eta = Eta
        self.num_features = num_features

    # training the data from readFile
    def train(self, trainExamples):
        for t in range(self.numIters):
            # print 'Starting iteration {}'.format(t)
            for i in range(len(trainExamples)):
                example = trainExamples[i]
                feature_vector = np.reshape(example[0], (self.num_features,) )
                label = np.nonzero(example[1])[0][0]
                scores = [np.dot(self.weights[k], feature_vector) for k in range(10)]
                classification = scores.index( max(scores) )
                if classification != label:
                    gradient = self.eta * feature_vector
                    self.weights[classification] = self.weights[classification] - gradient
                    self.weights[label] = self.weights[label] + gradient
        print 'Linear Classifier training done'
    
    # Using the weights from training, use it to classifiy a result
    # returns a tuple: (-1, digit) if not confident,
    # (1, digit) if confident
    def classify(self, example, constraint):
        results = [np.dot(self.weights[k], example[0]) for k in range(10)]
        sumresults = 0
        for result in results:
            if result > 0:
                sumresults += result
        best_result = max(results)
        if (best_result / float(sumresults)) < constraint:
            return (-1, results.index(best_result))
        return (1, results.index(best_result))

    # test if the classification was correct
    # output necessary data
    def test(self, testExamples):
        l = range(20, 66)
        l = l[0::3]
        for c in l:
            constraint = float(c) / 100.0
            print "constraint is now: " + str(constraint)
            total_correct = 0
            total_rejected = 0
            false_pos = 0 # wrong classification
            false_neg = 0 # rejected, but correct classification
            for i in range(len(testExamples)):
                example = testExamples[i]
                classification = self.classify(example, constraint)
                label = example[1]
                if classification[0] == 1: # ai was confident
                    if label == classification[1]:
                        total_correct = total_correct + 1
                    else:
                        false_pos += 1
                else:
                    if label == classification[1]:
                        false_neg += 1
                    total_rejected += 1
            print "accuracy: " + str (( float(total_correct)/(len(testExamples) - total_rejected) ))
            print "rejected: " + str(total_rejected)
            print "false_pos: " + str(false_pos)
            print "false_neg: " + str(false_neg)