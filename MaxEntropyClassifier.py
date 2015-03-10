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
    documents, words = InputParser.InputParserNoFrequencies("unlabeled.REVIEW", 'utf8')
    
    document_features = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in documents]
    
    train_set, test_set = document_features[400:], document_features[:400]
    
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter = 3)
    
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)

    
def ClassifierWithFreq():
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')

    document_features = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in documents]

    train_set, test_set = document_features[400:], document_features[:400]
    
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter = 3)
    
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
def ClassifierWithBestWords():
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')
    best_words = InformationGainCalculator.CalculateBestWords(documents)
    document_features = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in documents]
    
    train_set, test_set = document_features[400:], document_features[:400]
    
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter = 3)
    
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(10)

print("No Frequencies")   
ClassifierNoFreq()

print("Frequencies")
ClassifierWithFreq()

print("Best Words")
ClassifierWithBestWords()