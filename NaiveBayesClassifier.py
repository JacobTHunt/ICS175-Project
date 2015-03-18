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
    documents, words = InputParser.InputParserNoFrequencies("unlabeled.REVIEW", 'utf8')
    
    document_features = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in documents]
    
    train_set, test_set = document_features[400:], document_features[:400]
    
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
def ClassifierWithFrequencies():
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')

    document_features = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in documents]

    train_set, test_set = document_features[400:], document_features[:400]
    
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
def ClassifierWithBestWords():
    documents, words = InputParser.InputParserWithFrequencies("unlabeled.REVIEW", 'utf8')
    best_words = InformationGainCalculator.CalculateBestWords(documents)
    document_features = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in documents]
    
    train_set, test_set = document_features[400:], document_features[:400]
    
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print("Majority Classifier")
    maj_classifier = MajorityClassifier.MajorityClassifier(train_set)
    
    print("Majority Class Accuracy")
    print(maj_classifier.labels)
    print(maj_classifier.accuracy(test_set))
    
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)

print("No Frequencies")   
ClassifierNoFrequencies()

print("Frequencies")
ClassifierWithFrequencies()

print("Best Words")
ClassifierWithBestWords()