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
    documents, words = InputParser.InputParserNoFrequencies("unlabeled.REVIEW", 'utf8')
    
    document_features = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in documents]
    
    train_set, test_set = document_features[400:], document_features[:400]
    
    classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    
    print(nltk.classify.accuracy(classifier, test_set))
    
    
def ClassifierWithFreq():
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')

    document_features = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in documents]

    train_set, test_set = document_features[400:], document_features[:400]
    
    classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    
    print(nltk.classify.accuracy(classifier, test_set))
    
def ClassifierWithBestWords():
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')
    best_words = InformationGainCalculator.CalculateBestWords(documents)
    document_features = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in documents]
    
    train_set, test_set = document_features[400:], document_features[:400]
    
    classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    
    print(nltk.classify.accuracy(classifier, test_set))

print("No Frequencies")   
ClassifierNoFreq()

print("Frequencies")
ClassifierWithFreq()

print("Best Words")
ClassifierWithBestWords()