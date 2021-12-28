#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:48:53 2020

@author: jyotivashishth
"""

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC , LinearSVC , NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import codecs
import pickle
import pandas as pd
import os
from Reusable_Functions import *

#input path
_input_path = os.path.abspath(os.path.dirname('__file__'))
#file path
CSV_File_Path = os.path.join(_input_path , "Datasets")
#pickle paths
Pickle_Path = os.path.join(_input_path , "Pickled_files")

#read entire Data set
pdPdFrm_Test_Tweets = pd.read_csv(os.path.join(CSV_File_Path,"Test_Data.csv"),\
                                  encoding = "ISO-8859-1")

    
document_f = open(os.path.join(Pickle_Path,"documents.pickle"),"rb")
documents = pickle.load(document_f)
document_f.close()

word_Feature_F =open(os.path.join(Pickle_Path,"word_features5k.pickle"),"rb")
word_features = pickle.load(word_Feature_F)
word_Feature_F.close()



def find_features(document):
    words = word_tokenize(document)
    features ={}
    for w in word_features:
        features[w] = (w in words)
        
    return features


featureset = [(find_features(rev), category) for (rev, category ) in documents ]

random.shuffle(featureset )


#calculate for 70-30% train test split
Split_index = round(len(featureset) * 0.70)

#positive 
tarining_set = featureset[:Split_index]
testing_set = featureset[Split_index:]


#open Multibinomial NB pickle file
open_file = open(os.path.join(Pickle_Path,"MultinomialNB_classifier5k.pickle"),"rb")
MultinomialNB_Classifier = pickle.load(open_file)
open_file.close


#logistic LogisticRegression Classifier
open_file =open(os.path.join(Pickle_Path,"LogisticRegression_classifier.pickle"),"rb")
LR_Classifier = pickle.load(open_file)
open_file.close


#SGD_Classifier
open_file = open(os.path.join(Pickle_Path,"SGD_classifier.pickle"),"rb")
SGD_Classifier = pickle.load(open_file)
open_file.close


#AdaBoostClassifier with  LinearSVC
open_file = open(os.path.join(Pickle_Path,"AdaBoost_LinSVC_classifier.pickle"),"rb")
AdaBoost_LinSVC_Classifier = pickle.load(open_file)
open_file.close


#AdaBoostClassifier with  LogisticRegression
open_file = open(os.path.join(Pickle_Path,"AdaBoost_LR_classifier.pickle"),"rb")
AdaBoost_LR_Classifier = pickle.load(open_file)
open_file.close


Voted_Classifier = Best_Classifier(MultinomialNB_Classifier,LR_Classifier,\
                SGD_Classifier,AdaBoost_LinSVC_Classifier,AdaBoost_LR_Classifier)

def How_DO_I_FEEL(text):
    feats = find_features(text)
    return Voted_Classifier.confidence(feats) ,Voted_Classifier.classify(feats)