# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:00:34 2015

@author: jthunt
"""

import FeatureExtracter
import InformationGainCalculator
import InputParser
import nltk

def StarClassifierWithoutFreq():
    print("Loading Documents")
    train_documents, train_words = InputParser.InputParserNoFrequencies("train", "Latin-1")
    test_documents,  test_words  = InputParser.InputParserNoFrequencies("test",  "Latin-1")
    
    print("Extracting Features")
    train_set = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in train_documents]
    test_set  = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in test_documents ]
    
    print("Training Classifier")
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter = 3)
    
    print("Accuracy")
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)

def StarClassifierWithFreq():
    print("Loading Documents")
    train_documents, train_words = InputParser.InputParserWithFrequencies("train", "Latin-1")
    test_documents,  test_words  = InputParser.InputParserWithFrequencies("test",  "Latin-1")
    
    print("Extracting Features")
    train_set = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in train_documents]
    test_set  = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in test_documents]
    
    print("Training Classifier")
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter = 3)
    
    print("Accuracy")
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
def StarClassifierWithBestWords():
    print("Loading Documents")
    train_documents, train_words = InputParser.InputParserWithFrequencies("train", "Latin-1")
    test_documents,  test_words  = InputParser.InputParserWithFrequencies("test",  "Latin-1")
    
    print("Finding Best Words")
    best_words = InformationGainCalculator.CalculateBestWordsStarRating(train_documents)
    
    print("Extracting Features")
    train_set = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in train_documents]
    test_set  = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in test_documents ]
    
    print("Training Classifier")
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter = 10)
    
    print("Accuracy")
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
print("No Frequencies")
StarClassifierWithoutFreq()
print("")

print("With Frequencies")
StarClassifierWithFreq()
print("")

print("With Best Words")
StarClassifierWithBestWords()
print("")
