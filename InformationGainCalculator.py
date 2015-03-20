# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:29:13 2015

@author: Jacob
"""
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import operator

def CalculateBestWords(corpus):
    # Create frequency distributions for later
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    
    # For each document in the corpus
    for document in corpus:
        # Split out of the words from the label
        words = document[0]
        label = document[1]
        # For each word in the document
        for word in words:
            # Split off the word and frequency
            token, frequency = word.split(":")
            # Add the word to the distribution equal to the number of times it
            # occurs in the document
            for i in range(int(frequency)):
                word_fd[token.lower()] += 1
                label_word_fd[label][token.lower()] += 1
    
    # Figures out the number of words that apply to each label            
    pos_word_count = label_word_fd['positive'].N()
    neg_word_count = label_word_fd['negative'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    
    # This computes the probability that a word is in a given class, for each class
    for word, freq in word_fd.most_common(word_fd.N()):
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['positive'][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['negative'][word],
            (freq, neg_word_count), total_word_count)
            
        word_scores[word] = pos_score + neg_score
    
    # This sorts the list of words by their score and retrieves the 5000 best words
    best = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[:5000]
    best_words = set([w for w, s in best])
    return best_words
         
def CalculateBestWordsStarRating(corpus, number_of_words):
    # Create frequency distributions for later
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    
    # For each document in the corpus
    for document in corpus:
        # Split out of the words from the label
        words = document[0]
        label = document[1]
        # For each word in the document
        for word in words:
            # Split off the word and frequency
            token, frequency = word.split(":")
            # Add the word to the distribution equal to the number of times it
            # occurs in the document
            for i in range(int(frequency)):
                word_fd[token.lower()] += 1
                label_word_fd[label][token.lower()] += 1
                
    # Figure out the number of words that belong to each label
    one_word_count   = label_word_fd['1.0'].N()
    two_word_count   = label_word_fd['2.0'].N()
    four_word_count  = label_word_fd['4.0'].N()
    five_word_count  = label_word_fd['5.0'].N()
    total_word_count = one_word_count + two_word_count + four_word_count + five_word_count
        
    word_scores = {}
    
    # This computes the probability that a word is in a given class, for each class
    for word, freq in word_fd.most_common(word_fd.N()):
        #print(word)
        #print(label_word_fd['3.0'][word], freq, total_word_count)
        one_score = BigramAssocMeasures.chi_sq(label_word_fd['1.0'][word],
            (freq, one_word_count), total_word_count)
            
        two_score = BigramAssocMeasures.chi_sq(label_word_fd['2.0'][word],
            (freq, two_word_count), total_word_count)
            
        four_score = BigramAssocMeasures.chi_sq(label_word_fd['4.0'][word],
            (freq, four_word_count), total_word_count)
            
        five_score = BigramAssocMeasures.chi_sq(label_word_fd['5.0'][word],
            (freq, five_word_count), total_word_count)
            
        word_scores[word] = one_score + two_score + four_score + five_score
     
    # This sorts the list of words by their score and retrieves the number equal
    # to the parameter number_of_words
    best = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[:number_of_words]
    best_words = set([w for w, s in best])
    return best_words
    