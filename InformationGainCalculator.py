# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:29:13 2015

@author: Jacob
"""
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import operator

def CalculateBestWords(corpus):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    
    for document in corpus:
        words = document[0]
        label = document[1]
        for word in words:
            token, frequency = word.split(":")
            for i in range(int(frequency)):
                word_fd[token.lower()] += 1
                label_word_fd[label][token.lower()] += 1
                
    pos_word_count = label_word_fd['positive'].N()
    neg_word_count = label_word_fd['negative'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
     
    for word, freq in word_fd.most_common(word_fd.N()):
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['positive'][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['negative'][word],
            (freq, neg_word_count), total_word_count)
            
        word_scores[word] = pos_score + neg_score
     
    best = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[:5000]
    best_words = set([w for w, s in best])
    return best_words
         
def CalculateBestWordsStarRating(corpus):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    
    for document in corpus:
        words = document[0]
        label = document[1]
        for word in words:
            token, frequency = word.split(":")
            for i in range(int(frequency)):
                word_fd[token.lower()] += 1
                label_word_fd[label][token.lower()] += 1
                
    one_word_count   = label_word_fd['1.0'].N()
    two_word_count   = label_word_fd['2.0'].N()
    four_word_count  = label_word_fd['4.0'].N()
    five_word_count  = label_word_fd['5.0'].N()
    total_word_count = one_word_count + two_word_count + four_word_count + five_word_count
        
    word_scores = {}
    
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
     
    best = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)[:5000]
    best_words = set([w for w, s in best])
    return best_words
    
    
    """
    for word in movie_reviews.words(categories=['pos']):
        word_fd.inc(word.lower())
        label_word_fd['pos'].inc(word.lower())
     
    for word in movie_reviews.words(categories=['neg']):
        word_fd.inc(word.lower())
        label_word_fd['neg'].inc(word.lower())
     
    # n_ii = label_word_fd[label][word]
    # n_ix = word_fd[word]
    # n_xi = label_word_fd[label].N()
    # n_xx = label_word_fd.N()
     
    pos_word_count = label_word_fd['pos'].N()
    neg_word_count = label_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
     
    word_scores = {}
     
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
            (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
     
    best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
    bestwords = set([w for w, s in best])
    """