# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:31:35 2015

@author: jthunt
"""

def ExtractFeaturesNoFrequencies(document):    
    document_words = set(document) 
    features = {}
    for word in document_words:
        features['contains(%s)' % word] = True
    return features
    
def ExtractFeaturesWithFrequencies(document):
    document_words = set(document)
    
    features = {}
    for word in document_words:
        token, frequency = word.split(":")
        features['contains(%s)' % token] = True
        features['frequency(%s)' % token] = frequency
    return features
    
def ExtractFeaturesBestWords(document, best_words):
    document_words = set(document)
    
    features = {}
    for word in document_words:
        token, frequency = word.split(":")
        if token in best_words:
            features['contains(%s)' % token] = True
            
    return features