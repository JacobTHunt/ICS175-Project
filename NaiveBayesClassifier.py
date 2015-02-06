# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:54:08 2015

@author: jthunt
"""
import FeatureExtracter
import InputParser
import nltk

def ClassifierNoFrequencies():
    documents, words = InputParser.InputParserNoFrequencies("unlabeled.REVIEW")
    
    document_features = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in documents]
    
    train_set, test_set = document_features[400:], document_features[:400]
    
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
def ClassifierWithFrequencies():
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW")

    document_features = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in documents]

    train_set, test_set = document_features[400:], document_features[:400]
    
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)

print("No Frequencies")   
ClassifierNoFrequencies()

print("Frequencies")
ClassifierWithFrequencies()