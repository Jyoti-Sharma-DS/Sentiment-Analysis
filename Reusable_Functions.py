#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:13:18 2020

@author: jyotivashishth
"""



import nltk
from nltk.classify import ClassifierI
import pandas as pd
import os
from statistics import mode
from nltk.tokenize import word_tokenize
#variable declarrations
smiling_emo = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
crying_emo = r" (:'?[/|\(]) "


##Class for Votes 
class Best_Classifier(ClassifierI):
    
    def __init__(self , *classifiers):
        self._Classifiers = classifiers
        
    def classify(self, features ):
        vote =[]
        counter = 0
        for c in self._Classifiers:
            v = c.classify(features)
            # print("c")
            # print(c)
            # print("v")
            # print(v)
            counter+=1
            vote.append(v)
        # print("------------------------------------------------")
        # print("vote")
        # print(vote)
        # print("counter")
        # print(counter)
        return mode(vote)
    
    def confidence(self, features):
        votes =[]
        for c in self._Classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        
        return conf
            


#Function for clearing text
def transform( X , use_mention):
        # We can choose between keeping the mentions
        # or deleting them
        if use_mention:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")
        else:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")
            
        # Keeping only the word after the #
        X = X.str.strip()
        X = X.str.replace("#", "")
        X = X.str.replace(r"[-\.\n]", "")
        # Removing HTML garbage
        X = X.str.replace(r"&\w+;", "")
        # Removing links
        X = X.str.replace(r"https?://\S*", "")
        # replace repeated letters with only two occurences
        # heeeelllloooo => heelloo
        X = X.str.replace(r"(.)\1+", r"\1\1")
        # mark emoticons as happy or sad
        X = X.str.replace(smiling_emo, " happyemoticons ")
        X = X.str.replace(crying_emo, " sademoticons ")
        X = X.str.lower()
        return X


def find_features(document):
    print("---start---")
    words = word_tokenize(document)
    features ={}
    for w in word_features:
        features[w] = (w in words)
    print("----end-------")    
    print("---------------") 
    return features