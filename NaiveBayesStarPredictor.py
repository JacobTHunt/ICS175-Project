# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:11:45 2015

@author: Jacob
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
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print("Accuracy")
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)

def StarPredictorBestWords():
    print("Loading Documents")
    train_documents, train_words = InputParser.InputParserWithFrequencies("train", "Latin-1")
    test_documents,  test_words  = InputParser.InputParserWithFrequencies("test",  "Latin-1")
    
    print("Finding Best Words")
    best_words = InformationGainCalculator.CalculateBestWordsStarRating(train_documents)
    
    print("Extracting Features")
    train_set = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in train_documents]
    test_set  = [(FeatureExtracter.ExtractFeaturesBestWords(d, best_words), c) for (d,c) in test_documents ]
    
    document, label = test_set[0][0], test_set[0][1]
    
    print("Training Classifier")
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print("Accuracy")
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    
    prob_labels = classifier.prob_classify(document);
    
    print("Document Label: ", label)
    for sample in prob_labels.samples():
        print(sample, ": ", prob_labels.prob(sample))
    


print("Star Predictor No Frequencies")
#StarPredictorNoFrequencies()

print("Star Predictor Best Words") 
StarPredictorBestWords()