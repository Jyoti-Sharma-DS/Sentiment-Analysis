#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:04:39 2020

@author: jyotivashishth

purpose of  script :
    Test the test data set using load_pickled_mod
    
"""

import load_pickled_mod as lp
import pandas as pd
import os
from Reusable_Functions import *

#input path
_input_path = os.path.abspath(os.path.dirname('__file__'))
#file path
CSV_File_Path = os.path.join(_input_path , "Datasets")

#read entire Data set
pdPdFrm_Test_Tweets_1 = pd.read_csv(os.path.join(CSV_File_Path , "Test_Data.csv"),\
                                    encoding = "ISO-8859-1")

#list for storing sentiment and confidence
Sentiment_Predicted = []
confidence= []

print("Classification process triggered !!")
#loop in all Tweets
for x in range(0,len(pdPdFrm_Test_Tweets_1)):
    conf,sentiment = lp.How_DO_I_FEEL(pdPdFrm_Test_Tweets_1.loc[x,'SentimentText'])
    Sentiment_Predicted.append(sentiment)
    confidence.append(conf)
    
print("Classification process Completed !!")
accuracy_sentiment = sum(pdPdFrm_Test_Tweets_1['Sentiment']==Sentiment_Predicted)/len(pdPdFrm_Test_Tweets_1['Sentiment'])

#accuracy of voted classifier
print("Voted classifier accuracy :", str(round(accuracy_sentiment,3) * 100 ))