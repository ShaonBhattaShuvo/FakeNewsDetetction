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
#cheking the total number of words in the dataset
total_words = []
for row in dataset.required:
    for word in row:
        total_words.append(word)
len(total_words)
#lenght of unique words
unique_words=len(set(total_words))
print(unique_words)
#joining the words to create paragraphs
dataset['required_join'] = dataset['required'].apply(lambda word : " ".join(word))
print(dataset['required_join'][0])
#plotting barchart of number of news in each category
plt.figure(figsize=(8,5))
sns.countplot(y='subject',data=dataset)
#plotting barchart of number of true news and fake news
sns.countplot(y='isTrue',data=dataset)
#plotting wordcloud for true news
plt.figure(figsize=(12,8))
wc =  wordcloud.WordCloud(width=500, height=200,max_words=1000,
                          stopwords=stopWords).generate(" ".join(dataset[dataset.isTrue==1].required_join))
plt.imshow(wc,interpolation='bilinear')
#plotting wordcloud for fake news
plt.figure(figsize=(12,8))
wc =  wordcloud.WordCloud(width=500, height=200,max_words=1000,
                          stopwords=stopWords).generate(" ".join(dataset[dataset.isTrue==1].required_join))
plt.imshow(wc,interpolation='bilinear')
#counting maximum number of token size amoung all the documents
nltk.download('punkt')
maxLen = -1
for row in dataset.required_join:
    token = nltk.word_tokenize(row)
    if(len(token)>maxLen):
        maxLen=len(token)
print("Maximum number of words in a single doucment is: ",maxLen)
#Spliiting the dataset into train and testset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset.required_join,dataset.isTrue,test_size=0.2)
#Word Embedding
#Crating training sequences and test sequences
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=unique_words)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
print("The embedding for document: ",dataset.required_join[0], 
      "\n is: ",tokenizer.texts_to_sequences(dataset.required_join[0]) ) #this is how train and test sequences created
