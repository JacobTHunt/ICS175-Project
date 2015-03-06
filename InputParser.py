# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 11:36:47 2015

@author: jthunt
"""


"""
Reads in a REVIEW file and returns a list of documents along with their label
"""
import re


def InputParserNoFrequencies(file_name, encoding):    
    lines = [line.strip() for line in open(file_name, encoding=encoding)]
    
    documents = [line.split(" ") for line in lines]
    
    document_features = []
    words = set()
    
    for document in documents:
        category = re.sub(r'#label#:', '', document[-1])
        document = document[0:-1]
        
        for i in range(len(document)):
            document[i] = re.sub(r':[0-9]*', '', document[i])
            words.add(document[i])

        document_features.append((document, category))
        
    return [document_features, words]
    
def InputParserWithFrequencies(file_name, encoding):    
    lines = [line.strip() for line in open(file_name, encoding=encoding)]
    
    documents = [line.split(" ") for line in lines]
    words = set()
    document_features = []    
    
    for document in documents:
        category = re.sub(r'#label#:', '', document[-1])
        document = document[0:-1]

        document_features.append((document, category))

    return [document_features, words];