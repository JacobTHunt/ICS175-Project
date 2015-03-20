# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:56:04 2015

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
    
    # Create a classifier with the train set and algorithm 0
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter = 3)
    
    # Print the accuracy when predicting the test set and show the most informative features
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)

    
def ClassifierWithFreq():
    # Read in the documents, keeping the frequencies
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')

    # Extract the features from each document using the frequencies of the words as well
    document_features = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in documents]

    # Split the train and test set up
    train_set, test_set = document_features[400:], document_features[:400]
    
    # Create a classifier with the train set and algorithm 0
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter = 3)
    
    # Print the accuracy when predicting the test set and show the most informative features
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
def ClassifierWithBestWords():
    # Read in the documents, keeping the frequencies
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')
    
    # Calculate the best words using the chi_squared distribution
    best_words = InformationGainCalculator.CalculateBestWords(documents)
    
    # Extract features from the document using the best words
    document_features = [(FeatureExtracter.FeaturesEqualsBestWords(d, best_words), c) for (d,c) in documents]
    
    # Split the train and test set up
    train_set, test_set = document_features[400:], document_features[:400]
    
    # Create a classifier with the train set and algorithm 0
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter = 3)
    
    # Print the accuracy when predicting the test set and show the most informative features
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(10)

print("No Frequencies")   
ClassifierNoFreq()

print("Frequencies")
ClassifierWithFreq()

print("Best Words")
ClassifierWithBestWords()