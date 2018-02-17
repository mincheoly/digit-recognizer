import csv
import collections
import random
import numpy as np
from util import *

# this linear classifier contains a hyperparameter tuner that tunes to the best
# variable values: numiters and eta
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
        # print 'Linear Classifier training done'
    
    # Using the weights from training, use it to classifiy a result
    def classify(self, example):
        results = [np.dot(self.weights[k], example[0]) for k in range(10)]
        return results.index(max(results))

    # test if the classification was correct
    # returns the accuracy determined by the given numiters and eta
    def test(self, testExamples):
        total_correct = 0
        for i in range(len(testExamples)):
            example = testExamples[i]
            classification = self.classify(example)
            label = example[1]
            if label == classification:
                total_correct = total_correct + 1
        # print str( float(total_correct) / len(testExamples) )
        return float(total_correct) / len(testExamples)



