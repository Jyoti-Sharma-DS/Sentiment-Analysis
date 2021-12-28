#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:12:07 2020

@author: jyotivashishth
"""


from plotnine import *
import pandas as pd
import os
from Reusable_Functions import *

#input path
_input_path = os.path.abspath(os.path.dirname('__file__'))

CSV_File_Path = os.path.join(_input_path , "Datasets")

print(" Load the Train dataset")
#read entire Data set
pdPdFrm_Train_Tweets1 = pd.read_csv(os.path.join(CSV_File_Path , "Train_Data.csv"),encoding = "ISO-8859-1")
pdPdFrm_Train_Tweets1['SentimentText_old']  = pdPdFrm_Train_Tweets1['SentimentText'].copy()
pdPdFrm_Train_Tweets1['SentimentText'] = transform(pdPdFrm_Train_Tweets1['SentimentText'], "Y")
pdPdFrm_Train_Tweets = pdPdFrm_Train_Tweets1.copy()

#Calculate the length of all tweets
pdPdFrm_Train_Tweets['Length_of_Tweets'] = pdPdFrm_Train_Tweets['SentimentText'].str.len()


#show the length of tweets
my_plot = ggplot(data=pdPdFrm_Train_Tweets, mapping=aes(x='Sentiment' , fill='Sentiment' )) +\
    geom_bar(stat="bin", width=1)+\
    labs(title = "Tweets Distribution Train Dataset" , x = "Sentiment")
print(my_plot)


print("No of Positive tweets in Train dataset :") 
print(len(pdPdFrm_Train_Tweets.query("Sentiment=='Positive'")))
print("No of Negative tweets in Train dataset :")
print(len(pdPdFrm_Train_Tweets.query("Sentiment=='Negative'")))


print("Calculate the % of positive and negative Tweet texts..")
X = ["Positive" , "Negative"]
Percentage =[]
Percentage.append(round(len(pdPdFrm_Train_Tweets.query("Sentiment=='Positive'"))/\
                        len(pdPdFrm_Train_Tweets) * 100 ,3))

Percentage.append(round(len(pdPdFrm_Train_Tweets.query("Sentiment=='Negative'"))/\
                        len(pdPdFrm_Train_Tweets) * 100 ,3))
    
PD_Train_per = pd.DataFrame({"Sentiment":X , "Percentage":Percentage}, columns=['Sentiment', "Percentage"])

my_plot = ggplot(data=PD_Train_per, mapping=aes(x='Sentiment', fill ="Percentage"  )) +\
    geom_col (aes(y="Percentage" ) )+\
    labs(title = "Sentiment % Distribution Train Dataset ",\
         x = "Sentiment" , y = "Percentage")+\
    geom_text(aes(y="Percentage", label="Percentage"))
print(my_plot)



print("Load the Test dataset")

#read entire Data set
pdPdFrm_Test_Tweets = pd.read_csv(os.path.join(CSV_File_Path , "Test_Data.csv"),encoding = "ISO-8859-1")


pdPdFrm_Test_Tweets['Length_of_Tweets'] = pdPdFrm_Test_Tweets['SentimentText'].str.len()


my_plot = ggplot(data=pdPdFrm_Test_Tweets, mapping=aes(x='Sentiment' , fill='Sentiment' )) +\
    geom_bar(stat="bin", width=1)+\
    labs(title = "Tweets Distribution Test Dataset" , x = "Sentiment")
print(my_plot)


print("No of Positive tweets :") 
print(len(pdPdFrm_Test_Tweets.query("Sentiment=='Positive'")))
print("No of Negative tweets :")
print(len(pdPdFrm_Test_Tweets.query("Sentiment=='Negative'")))


print("Understanding the data ")
frames =  [pdPdFrm_Train_Tweets, pdPdFrm_Test_Tweets]
#complete data frame
Complete_DataFrame = pd.concat(frames)

average_len = sum(Complete_DataFrame["Length_of_Tweets"])/len(Complete_DataFrame["Length_of_Tweets"])

print("Average length of tweets :")
print(average_len)

my_plot = ggplot(data=Complete_DataFrame, mapping=aes(x='Length_of_Tweets')) +\
    geom_histogram(color='white', bins=100 , fill="steelblue")+\
    labs(title = "Length of all Tweets" , x = "Length of Tweets")
print(my_plot)


Percentage =[]
Percentage.append(round(len(pdPdFrm_Test_Tweets.query("Sentiment=='Positive'"))/\
                        len(pdPdFrm_Test_Tweets) * 100 ,3))

Percentage.append(round(len(pdPdFrm_Test_Tweets.query("Sentiment=='Negative'"))/\
                        len(pdPdFrm_Test_Tweets) * 100 ,3))
    
PD_Train_per = pd.DataFrame({"Sentiment":X , "Percentage":Percentage}, columns=['Sentiment', "Percentage"])

my_plot = ggplot(data=PD_Train_per, mapping=aes(x='Sentiment', fill ="Percentage"  )) +\
    geom_col (aes(y="Percentage" ) )+\
    labs(title = "Sentiment % Distribution Test Dataset ",\
         x = "Sentiment" , y = "Percentage")+\
    geom_text(aes(y="Percentage", label="Percentage"))
print(my_plot)


print("Top 25 used words ....................")

for key in all_words:
   words.append(key) 
   freqs.append(all_words[key])

X_val = words[:50]
Y_val = freqs[:50]

pd_frame_val = pd.DataFrame({"Words":X_val, "Frequency":Y_val},
                            columns=["Words", "Frequency"])

my_plot = ggplot(data=pd_frame_val, mapping=aes(x='Words' )) +\
    geom_col (aes(y="Frequency" ) , fill="steelblue")+\
    labs(title = "Word Frequency Distribution(Allowed distribution - adjective,adverb,Noun)",\
         x = "Words")+\
    theme(axis_text_x=element_text(rotation=90, hjust=1))
print(my_plot)


#accuracy Plot
Classifier_names = ['﻿MultinomialNB','﻿Logistic Regression','﻿SGD',
'﻿AdaBoost (Linear SVC)','﻿AdaBoostClassifier (LogisticRegression)',
'﻿Voted Classifier(Train)' , '﻿Voted Classifier(Test)']

Accuracy_percentage = [ 73.298,
73.291,
73.257,
71.68,
72.017,
73.918,
71.899]

pd_frame_val = pd.DataFrame({"Classifier":Classifier_names,\
"AccuracyPercentage":Accuracy_percentage},\
                    columns=["Classifier", "AccuracyPercentage"])

my_plot = ggplot(data=pd_frame_val, mapping=aes(x='Classifier' )) +\
    geom_col (aes(y="AccuracyPercentage" ) , fill="steelblue")+\
    labs(title = "Classifier accuracies (Allowed words - adjective,adverb,Noun)",\
         x = "Classifiers")+\
    geom_text(aes(y="AccuracyPercentage", label="AccuracyPercentage"))+\
    theme(axis_text_x=element_text(rotation=90, hjust=1))
print(my_plot)




print("Top 25 used words ....................")
words=[]
freqs=[]
for key in all_words:
   words.append(key) 
   freqs.append(all_words[key])

X_val = words[:50]
Y_val = freqs[:50]

pd_frame_val = pd.DataFrame({"Words":X_val, "Frequency":Y_val},
                            columns=["Words", "Frequency"])

my_plot = ggplot(data=pd_frame_val, mapping=aes(x='Words' )) +\
    geom_col (aes(y="Frequency" ) , fill="steelblue")+\
    labs(title = "Word Frequency Distribution(Allowed distribution - adjective,adverb)",\
         x = "Words")+\
    theme(axis_text_x=element_text(rotation=90, hjust=1))
print(my_plot)



#accuracy Plot
Classifier_names = ['1-﻿MultinomialNB','2-﻿Logistic Regression','3-﻿SGD',
'4-﻿AdaBoost (Linear SVC)','﻿5-AdaBoostClassifier (LogisticRegression)',
'6-﻿Voted Classifier(Train)' , '7-﻿Voted Classifier(Test)']

Accuracy_percentage = [ 73.11,73.44, 73.34 ,71.47,70.91,73.56,71.89]

pd_frame_val = pd.DataFrame({"Classifier":Classifier_names,\
"AccuracyPercentage":Accuracy_percentage},\
                    columns=["Classifier", "AccuracyPercentage"])

