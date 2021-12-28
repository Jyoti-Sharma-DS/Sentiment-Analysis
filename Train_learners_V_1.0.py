#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:26:35 2020

@author: jyotivashishth

Learner script

purpose of  script :
    - create documen and feature sets 
    - Train classifiers and save them in 'Pickled_files'
    
Config :
    - To create feature sets allowing Verb ,  adjective and noun 
    set allowed words as below(line 62)
    allowed_word_types =["J" , "V" , "RB" , "N"]
    - To create feature sets allowing Verb  and adjectives
    allowed_word_types =["J" , "V" , "RB" ]

"""
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import  LinearSVC 
from sklearn.ensemble import AdaBoostClassifier
from nltk.classify import WekaClassifier
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import codecs
import pickle
import os
import pandas as pd
import datetime
from Reusable_Functions import *

#input path
_input_path = os.path.abspath(os.path.dirname('__file__'))

CSV_File_Path = os.path.join(_input_path , "Datasets")

Pickle_Path = os.path.join(_input_path , "Pickled_files")

#read entire Data set
pdPdFrm_Train_Tweets1 = pd.read_csv(os.path.join(CSV_File_Path , "Train_Data.csv"),encoding = "ISO-8859-1")
pdPdFrm_Train_Tweets1['SentimentText_old']  = pdPdFrm_Train_Tweets1['SentimentText'].copy()
pdPdFrm_Train_Tweets1['SentimentText'] = transform(pdPdFrm_Train_Tweets1['SentimentText'], "Y")
pdPdFrm_Train_Tweets = pdPdFrm_Train_Tweets1.copy()


#print documents and words
documents=[]
all_words =[]  

print(datetime.datetime.now())

#Define allowed words - adjective , adverb , verb
allowed_word_types =["J" , "V" , "RB" ]


print("Allowed words : Adjective and Verb")

print("Document creation triggered !!")

#loop in and create documents and filtering words for creating feature set 
for x in range(0, len(pdPdFrm_Train_Tweets)):
    # print(x)
    line = pdPdFrm_Train_Tweets.loc[x,'SentimentText']
    sentiment =  pdPdFrm_Train_Tweets.loc[x,'Sentiment']
    documents.append((line,sentiment))
    #tokenize words
    words = word_tokenize(line)
    #tag the words as noun adjective 
    pos = nltk.pos_tag(words)
    for w in pos: 
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

print("Document creation completed !!")
     

#pickle 
save_documents = open(os.path.join(Pickle_Path ,"documents.pickle"),"wb")
pickle.dump(documents, save_documents)
save_documents.close()

    
all_words = nltk.FreqDist(all_words)


# print(all_words.most_common(10))
# print(all_words["stupid"])


print("word_features creation started !!")

word_features = list(all_words.keys())[:5000]

#pickle the features 
save_word_ft = open(os.path.join(Pickle_Path ,"word_features5k.pickle"),"wb")
pickle.dump(word_features, save_word_ft)
save_word_ft.close()


def find_features(document):
    # print("---start---")
    words = word_tokenize(document)
    features ={}
    for w in word_features:
        features[w] = (w in words)
    # print("----end-------")    
    # print("---------------") 
    return features

featureset = [(find_features(rev), category) for (rev, category ) in documents ]

print("word_features creation Completed !!")

random.seed(60)
random.shuffle(featureset)
print(len(featureset))

#calculate for 70-30% train test split
Split_index = round(len(featureset) * 0.70)
      
#positive 
training_set = featureset[:Split_index]
testing_set = featureset[Split_index:]


#train MultinomialNB
print("Train MultinomialNB Classifier....")
MultinomialNB_Classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_Classifier.train(training_set)

print("MultinomialNB_Classifier accuracy % : " ,\
      (nltk.classify.accuracy(MultinomialNB_Classifier, testing_set)) *100)
print("Saving MultinomialNB Classifier....")
save_classifier = open(os.path.join(Pickle_Path ,"MultinomialNB_classifier5k.pickle"),"wb")
pickle.dump(MultinomialNB_Classifier, save_classifier)
save_classifier.close()


#train LogisticRegression
print("Train LogisticRegression Classifier....")
LR_Classifier = SklearnClassifier(LogisticRegression(solver='newton-cg'))
LR_Classifier.train(training_set)
print("LR_Classifier accuracy % : ",(nltk.classify.accuracy(LR_Classifier,\
testing_set)) *100)
print("Saving LogisticRegression Classifier....")    
save_classifier = open(os.path.join(Pickle_Path ,"LogisticRegression_classifier.pickle"),"wb")
pickle.dump(LR_Classifier, save_classifier)
save_classifier.close()


#train LogisticRegression
print("Train SGD Classifier ....")
SGD_Classifier = SklearnClassifier(SGDClassifier())
SGD_Classifier.train(training_set)
print("SGD_Classifier accuracy % : ",(nltk.classify.accuracy(SGD_Classifier,\
testing_set)) *100)
print("Saving SGD Classifier ....")
save_classifier = open(os.path.join(Pickle_Path ,"SGD_classifier.pickle"),"wb")
pickle.dump(SGD_Classifier, save_classifier)
save_classifier.close()


#train AdaBoostClassifier with  LinearSVC
print("Train AdaBoostClassifier Classifier with  LinearSVC....")
LinearSVC_Classifier = LinearSVC()
AdaBoost_LinSVC_Classifier = SklearnClassifier(AdaBoostClassifier(LinearSVC_Classifier,\
                                                                 algorithm='SAMME'))
AdaBoost_LinSVC_Classifier.train(training_set)
print("AdaBoost_LinSVC_Classifier accuracy % : ",\
      (nltk.classify.accuracy(AdaBoost_LinSVC_Classifier,\
 testing_set)) *100)
print("Saving AdaBoostClassifier Classifier with  LinearSVC....")                                                            
save_classifier = open(os.path.join(Pickle_Path ,"AdaBoost_LinSVC_classifier.pickle"),"wb")
pickle.dump(AdaBoost_LinSVC_Classifier, save_classifier)
save_classifier.close()

#train AdaBoostClassifier with  LogisticRegression
print("Train AdaBoostClassifier Classifier with  LogisticRegression....")
LR = LogisticRegression(solver='newton-cg')
AdaBoost_LR_Classifier = SklearnClassifier(AdaBoostClassifier(LR, algorithm='SAMME'))
AdaBoost_LR_Classifier.train(training_set)
print("AdaBoost_Classifier accuracy % : ",(nltk.classify.accuracy(AdaBoost_LR_Classifier,\
 testing_set)) *100)  
print("Saving AdaBoostClassifier Classifier with  LogisticRegression....")                                                        
save_classifier = open(os.path.join(Pickle_Path ,"AdaBoost_LR_classifier.pickle"),"wb")
pickle.dump(AdaBoost_LR_Classifier, save_classifier)
save_classifier.close()

print(datetime.datetime.now())
Voted_Classifier = Best_Classifier(MultinomialNB_Classifier,LR_Classifier,\
                SGD_Classifier,AdaBoost_LinSVC_Classifier ,AdaBoost_LR_Classifier)


print("Voted_Classifier accuracy % : ",\
      (nltk.classify.accuracy(Voted_Classifier, testing_set))*100)