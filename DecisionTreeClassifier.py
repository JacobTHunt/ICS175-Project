# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:59:00 2015

@author: Edmond
"""

import FeatureExtracter
import InformationGainCalculator
import InputParser
import nltk

def ClassifierNoFreq():
    # Read in the documents, discarding frequencies
    documents, words = InputParser.InputParserNoFrequencies("unlabeled.REVIEW", 'utf8')
    
    # Extract the features from each document
    document_features = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in documents]
    
    # Split the train and test set up
    train_set, test_set = document_features[400:], document_features[:400]
    
    # Create a classifier with the train set
    classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    
    # Print the accuracy when predicting the test set
    print(nltk.classify.accuracy(classifier, test_set))
    
    
def ClassifierWithFreq():
    # read in the documents, keeping the frequencies
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')

    # Extract the features from each document using the frequencies of the words as well
    document_features = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in documents]

    # Split the train and test set up
    train_set, test_set = document_features[400:], document_features[:400]
    
    # Create a classifier with the train set
    classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    
    # Print the accuracy when predicting the test set
    print(nltk.classify.accuracy(classifier, test_set))
    
def ClassifierWithBestWords():
    # Extract the features from each document keeping the frequencies
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')
    
    # Calculate the best words to use baised on chi_squared distribution
    best_words = InformationGainCalculator.CalculateBestWords(documents)
    
    # Extract the features based on the best words found previously
    document_features = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in documents]
    
    # Split the train and test set up
    train_set, test_set = document_features[400:], document_features[:400]
    
    # Create a classifier with the train set
    classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    
    # Print the accuracy when predicting the test set
    print(nltk.classify.accuracy(classifier, test_set))

print("No Frequencies")   
ClassifierNoFreq()

print("Frequencies")
ClassifierWithFreq()

print("Best Words")
ClassifierWithBestWords()