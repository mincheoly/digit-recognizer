import numpy as np
import csv
import random
"""
Function name: readFile(filename)
This function takes an excel spreadsheet of data and converts them into a 
(target_vector, np.array of pixel features)
"""
def readFile(filename, mode):
    features_list = []
    labels_list = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if row[0] != 'label':
                classification = int(float(row[0]))
                label = np.zeros((10, 1))
                label[classification] = 1.0
                raw_pixels = map(int, row[1:len(row)])
                pixels = (1.0/255)*( np.reshape(np.array(raw_pixels, dtype='f'), ( len(raw_pixels), 1) ) )
                features_list.append(pixels)
                if mode == 'train':
                    labels_list.append(label)
                elif mode == 'test':
                    labels_list.append(classification)
    print 'Reading file done'
    return zip(features_list, labels_list)