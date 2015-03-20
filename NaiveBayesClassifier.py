# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:54:08 2015

@author: jthunt
"""
import FeatureExtracter
import InformationGainCalculator
import InputParser
import MajorityClassifier
import nltk

def ClassifierNoFrequencies():
    # Read in the documents, discarding frequencies
    documents, words = InputParser.InputParserNoFrequencies("unlabeled.REVIEW", 'utf8')
    
    # Extract the features from each document
    document_features = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in documents]
    
    # Split the train and test set up
    train_set, test_set = document_features[400:], document_features[:400]
    
    # Create a classifier with the train set
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    # Print the accuracy when predicting the test set and show the most informative features
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
def ClassifierWithFrequencies():
    # Read in the documents, keeping the frequencies
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')

    # Extract the features from each document using the frequencies of the words as well
    document_features = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in documents]
    
    # Split the train and test set up
    train_set, test_set = document_features[400:], document_features[:400]
    
    # Create a classifier with the train set
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    # Print the accuracy when predicting the test set and show the most informative features
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
def ClassifierWithBestWords():
    # Read in the documents, keeping the frequencies
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')
    best_words = InformationGainCalculator.CalculateBestWords(documents)
    document_features = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in documents]
    
    # Split the train and test set up
    train_set, test_set = document_features[400:], document_features[:400]
    
    # Create a classifier with the train set
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print("Majority Classifier")
    maj_classifier = MajorityClassifier.MajorityClassifier(train_set)
    
    print("Majority Class Accuracy")
    print(maj_classifier.labels)
    print(maj_classifier.accuracy(test_set))
    
    # Create a classifier with the train set and algorithm 0
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)

print("No Frequencies")   
ClassifierNoFrequencies()

print("Frequencies")
ClassifierWithFrequencies()

print("Best Words")
ClassifierWithBestWords()