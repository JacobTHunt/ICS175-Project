# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:52:38 2015

@author: Jacob
"""

import operator

class MajorityClassifier:
    labels = {}
    predict_label = 0;
    
    def __init__(self, train_set):
        for features, label in train_set:
            if label not in self.labels:
                self.labels[label] = 1
            else:
                self.labels[label] += 1
        
        self.predict_label = sorted(self.labels.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        
    
    def accuracy(self, test_set):
        num_docs = len(test_set)
        correct = 0.0
        for features, label in test_set:
            if label == self.predict_label:
                correct += 1
        
        return correct / num_docs
        