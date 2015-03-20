# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:11:45 2015

@author: Jacob
"""

import FeatureExtracter
import InformationGainCalculator
import InputParser
import MajorityClassifier
import nltk

def StarPredictorWithoutFrequencies():
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
    print("")
    
def StarPredictorWithFrequencies():
    print("Loading Documents")
    train_documents, train_words = InputParser.InputParserWithFrequencies("train", "Latin-1")
    test_documents,  test_words  = InputParser.InputParserWithFrequencies("test",  "Latin-1")
    
    print("Extracting Features")
    train_set = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in train_documents]
    test_set  = [(FeatureExtracter.ExtractFeaturesWithFrequencies(d), c) for (d,c) in test_documents ]
    
    print("Training Classifier")
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print("Accuracy")
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    print("")

def StarPredictorBestWords():
    print("Loading Documents")
    train_documents, train_words = InputParser.InputParserWithFrequencies("train", "Latin-1")
    test_documents,  test_words  = InputParser.InputParserWithFrequencies("all_balanced.review",  "Latin-1")
    
    print("Finding Best Words")
    best_words = InformationGainCalculator.CalculateBestWordsStarRating(train_documents, 4000)
    
    print("Extracting Features")
    train_set = [(FeatureExtracter.FeaturesEqualsBestWords(d, best_words), c) for (d,c) in train_documents]
    test_set  = [(FeatureExtracter.FeaturesEqualsBestWords(d, best_words), c) for (d,c) in test_documents ]
    
    print("Training Classifier")
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print("Majority Classifier")
    maj_classifier = MajorityClassifier.MajorityClassifier(train_set)
    
    print("Majority Class Accuracy")
    print(maj_classifier.labels)
    print(maj_classifier.accuracy(test_set))
    
    print("Accuracy")
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    print("")
    
    print("Positive/Negative Accuracy")
    print(StarToPositiveNegativeAccuracy(classifier, test_set))
    
def FindBestNumberOfWords():
    print("Loading Documents")
    train_documents, train_words = InputParser.InputParserWithFrequencies("test", "Latin-1")
    test_documents,  test_words  = InputParser.InputParserWithFrequencies("all_balanced.review",  "Latin-1")
    
    best_accuracy = 0;
    best_num_words = 0;
    
    for num_words in range(3500, 4500, 100):
        print("Finding Best Words with num_words: ", num_words)
        best_words = InformationGainCalculator.CalculateBestWordsStarRating(train_documents, num_words)
        
        print("Extracting Features")
        train_set = [(FeatureExtracter.FeaturesEqualsBestWords(d, best_words), c) for (d,c) in train_documents]
        test_set  = [(FeatureExtracter.FeaturesEqualsBestWords(d, best_words), c) for (d,c) in test_documents ]
        
        print("Training Classifier")
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        
        accuracy = nltk.classify.accuracy(classifier, test_set)
        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_num_words = num_words
            
    
    print(best_accuracy, best_num_words)
    
def StarToPositiveNegativeAccuracy(classifier, test_set):
    correct = 0.0
    total = len(test_set)
    
    for features, label in test_set:
        predicted_label = classifier.classify(features)
        
        if label == '1.0' or label == '2.0':
            if predicted_label == '1.0' or predicted_label == '2.0':
                correct += 1
            else:
                #print("Correct Label: ", label, "Predicted Label: ", predicted_label)
                prob_dist = classifier.prob_classify(features)
                string = ""
                for sample in prob_dist.samples():
                    string += sample + " " + str(prob_dist.prob(sample)) + " | "
                
                #print(string)
                #print(features)
        
        if label == '4.0' or label == '5.0':
            if predicted_label == '4.0' or predicted_label == '5.0':
                correct += 1
            else:
                #print("Correct Label: ", label, "Predicted Label: ", predicted_label)
                prob_dist = classifier.prob_classify(features)
                string = ""
                for sample in prob_dist.samples():
                    string += sample + " " + str(prob_dist.prob(sample)) + " | "
                
                #print(string)
                #print(features)
    
    return correct / total
    


print("Star Predictor Without Frequencies")
StarPredictorWithoutFrequencies()
print("")

print("Star Predictor With Frequencies")
StarPredictorWithFrequencies()
print("")

print("Star Predictor Best Words") 
StarPredictorBestWords()
print("")

#print("Find Best Number of Words")
#FindBestNumberOfWords()