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
        # Count the number of documents that apply to each label
        for features, label in train_set:
            if label not in self.labels:
                self.labels[label] = 1
            else:
                self.labels[label] += 1
        
        # Find the label with the most documents and remember it
        self.predict_label = sorted(self.labels.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        
    
    def accuracy(self, test_set):
        num_docs = len(test_set)
        correct = 0.0
        # For each document, if the label equals the largest label then it is correct
        for features, label in test_set:
            if label == self.predict_label:
                correct += 1
        
        return correct / num_docs
        