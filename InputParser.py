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
    # Reads the entire file in lines and strip the end line characters
    lines = [line.strip() for line in open(file_name, encoding=encoding)]
    
    # Splits each line based on quotation marks, each row is not an array of
    # words that corresponds to documents
    documents = [line.split(" ") for line in lines]
    
    document_features = []
    words = set()
    
    # For each document 
    for document in documents:
        # Splits off the label from the document
        category = re.sub(r'#label#:', '', document[-1])
        document = document[0:-1]
        
        # this removes the frequencies from the document, leaving only the
        # words
        for i in range(len(document)):
            document[i] = re.sub(r':[0-9]*', '', document[i])
            words.add(document[i])

        document_features.append((document, category))
        
    return [document_features, words]
    
def InputParserWithFrequencies(file_name, encoding):
    # Reads the entire file in lines and strip the end line characters
    lines = [line.strip() for line in open(file_name, encoding=encoding)]
    
    # Splits each line based on quotation marks, each row is not an array of
    # words that corresponds to documents
    documents = [line.split(" ") for line in lines]
    words = set()
    document_features = []    
    
    # For each document
    for document in documents:
        # Split the label of the document off
        category = re.sub(r'#label#:', '', document[-1])
        document = document[0:-1]

        # Here we are keeping the frequency so we don't get rid of it
        document_features.append((document, category))

    return [document_features, words];