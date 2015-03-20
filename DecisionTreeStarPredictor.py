# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:22:08 2015

@author: Edmond
"""

import FeatureExtracter
import InformationGainCalculator
import InputParser
import nltk

def StarPredictorNoFrequencies():
    print("Loading Documents")
    train_documents, train_words = InputParser.InputParserNoFrequencies("train", "Latin-1")
    test_documents,  test_words  = InputParser.InputParserNoFrequencies("test",  "Latin-1")
    
    print("Extracting Features")
    train_set = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in train_documents]
    test_set  = [(FeatureExtracter.ExtractFeaturesNoFrequencies(d), c) for (d,c) in test_documents ]
    
    print("Training Classifier")
    classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    
    print("Accuracy")
    print(nltk.classify.accuracy(classifier, test_set))

def StarPredictorBestWords():
    print("Loading Documents")
    train_documents, train_words = InputParser.InputParserWithFrequencies("train", "Latin-1")
    test_documents,  test_words  = InputParser.InputParserWithFrequencies("test",  "Latin-1")
    
    print("Finding Best Words")
    best_words = InformationGainCalculator.CalculateBestWordsStarRating(train_documents)
    
    print("Extracting Features")
    train_set = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in train_documents]
    test_set  = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in test_documents ]
    
    print("Training Classifier")
    classifier = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
    
    print("Accuracy")
    print(nltk.classify.accuracy(classifier, test_set))
    


print("Star Predictor No Frequencies")
StarPredictorNoFrequencies()

print("Star Predictor Best Words") 
StarPredictorBestWords()