# -*- coding: utf-8 -*-
"""
Shaon Bhatta Shuvo
"""
#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
#deleting rows to reduce the size (may be needed for quick test as full dataset training may take long time)
#fake_news.drop(fake_news.index[500:], inplace=True)
#true_news.drop(true_news.index[500:], inplace=True)

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
#Plotting the histogram for length of each doucment
plt.hist(x=[nltk.word_tokenize(doc) for doc in dataset.required_join],bins=100)
plt.show()
#Spliiting the dataset into train and testset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset.required_join,dataset.isTrue,test_size=0.2)#Creating validation set  by copying last 10 elements from the training set
#Creating validation set
X_val = X_train[30000:]
y_val = y_train[30000:]
#Removing the validation set from training set
X_train = X_train[:30000]
y_train = y_train[:30000]
#plotting the number of true news and fake news in traing, test and validation set
plt.hist(y_train)
plt.hist(y_test)
plt.hist(y_val)
#Word Embedding (Mapping word to vectors of real numbers)
#Crating training sequences and test sequences
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=unique_words)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
val_sequences = tokenizer.texts_to_sequences(X_val)
print("The embedding for document: ",dataset.required_join[0], 
      "\n is: ",tokenizer.texts_to_sequences(dataset.required_join[0]) ) #this is how train and test sequences created
#padding the sequnces (adding 0 to the end ) to make each sequence length exactly equal to maxLen
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_sequences_pad = pad_sequences(train_sequences, maxlen=maxLen, padding='post', truncating = 'post')
test_sequences_pad = pad_sequences(test_sequences, maxlen=maxLen, padding= 'post', truncating = 'post')
val_sequences_pad = pad_sequences(val_sequences, maxlen=maxLen, padding= 'post', truncating = 'post' )
#printing past three values to check padding whether the padding has been done correclty or not
for i,doc in enumerate(train_sequences_pad[:3]):
    print("The padded incoding for document ",i+1, " is: ",doc)

#Creating LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
model = Sequential()
model.add(Embedding(unique_words, output_dim=128))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
training=model.fit(train_sequences_pad,y_train,validation_data=(val_sequences_pad,y_val),batch_size=32, epochs=3)

#Visulaizing the Training and Validation Sets Loss and Accuracy
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
#Plot training and validation accuracy values
#axes[0].set_ylim(0,1) #if we want to limit axis in certain range
axes[0].plot(training.history['acc'], label='Train')
axes[0].plot(training.history['val_acc'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
#Plot training and validation loss values
#axes[1].set_ylim(0,1)
axes[1].plot(training.history['loss'], label='Train')
axes[1].plot(training.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
plt.tight_layout()
plt.show()

#performance evaluatoin
y_pred = model.predict(test_sequences_pad)
prediction = []
for i in range(len(y_pred)):
    if(y_pred[i].item()>0.5):
        prediction.append(1)
    else:
        prediction.append(0)
# Generating confusion matrics, details classification report
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
cm = confusion_matrix(list(y_test),prediction)
print("Confusion Matrix for Neural Network Model:\n ",cm)
print( "{0}".format(classification_report(list(y_test),prediction)))
# Generating accuracy in %, 
# Similary precision_score and recall_score can be used to generate precision and recall seperately
accuracy_test = accuracy_score(list(y_test),prediction)*100
print('Accuracy:%.2f' % accuracy_test,"%")
