# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:31:35 2015

@author: jthunt
"""

def ExtractFeaturesNoFrequencies(document):
    # Create a new set of the words for fast searching
    document_words = set(document) 
    features = {}
    # For each word in the document
    for word in document_words:
        # Add a feature that says that word is in the document
        features['contains(%s)' % word] = True
    return features
    
def ExtractFeaturesWithFrequencies(document):
    document_words = set(document)      
    features = {}
    # For each word in the document
    for word in document_words:
        # Split the word from the frequency
        token, frequency = word.split(":")
        # Create a feature that says the word is in the document
        features['contains(%s)' % token] = True
        # Create a feature that says how much a word occurs in the document
        features['frequency(%s)' % token] = frequency
        
    return features
    
def ExtractFeaturesBestWords(document, best_words):
    document_words = set(document)
    
    features = {}
    # For each word in the document
    for word in document_words:
        # Split the token and frequency apart, but only the token will be used
        token, frequency = word.split(":")
        # If the token is in the list of best words
        if token in best_words:
            # Add a feature saying that best word is in the document
            features['contains(%s)' % token] = True
            
def FeaturesEqualsBestWords(document, best_words):
    document_words = set(document)
    
    features = {}
    # For each word in the document
    for word in document_words:
        # Split the token and frequency apart, but only the token will be used
        token, frequency = word.split(":")
        # If the token is in the list of best words
        if token in best_words:
            # Add a feature saying that best word is in the document
            features['contains(%s)' % token] = True
        
    # For each word in the the list of best words
    for word in best_words:
        feature = 'contains(%s)' % word
        # If the document does not contain the word
        if feature not in features:
            # Create a feature saying that the word is NOT in the document
            features[feature] = False
            
    return features