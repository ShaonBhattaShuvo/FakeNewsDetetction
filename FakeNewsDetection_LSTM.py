# -*- coding: utf-8 -*-
"""
Shaon Bhatta Shuvo
"""
#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import nltk
import gensim
import wordcloud

#dataset loading
fake_news = pd.read_csv("Datasets/Fake.csv")
true_news = pd.read_csv("Datasets/True.csv")
#data info
fake_news.info()
true_news.info()
#checking the null values in the dataset(not required as we can see from info())
fake_news.isnull().sum()
true_news.isnull().sum()
#adding lebel to the dataset isTrue =1 for true and 0 for fake
fake_news["isTrue"]=0
true_news["isTrue"]=1
#creating final dataset
dataset = pd.concat([true_news,fake_news]).reset_index(drop=True)
#deleting the date column as we do not need that
dataset.drop(columns=['date'], inplace=True) #update change in dataset location
#merging title and text together so that we need not to check to differnt column
dataset['title_text']=dataset['title']+' '+dataset['text']
print(dataset['title_text'][0])
#stopwords identificatoin
nltk.download('stopwords')
from nltk.corpus import stopwords
stopWords = stopwords.words('English')
print(stopWords)
stopWords.extend(['edu','subject']) #adding some more words manyally to the list
#preprocessing the texts
def preprocessing(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token)>3 and token not in stopWords : 
            result.append(token)

    return result
#creating a required column which contains required words after preprocessing the dataset 
dataset['required'] = dataset['title_text'].apply(preprocessing)
print(dataset['required'][0])








