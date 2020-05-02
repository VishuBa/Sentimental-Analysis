#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import string
import nltk

#loading training and test sets
train=pd.read_csv("train_E6oV3lV.csv")
test=pd.read_csv("test_tweets_anuFYb8.csv")


#combining training and test sets for ease of preprocessing and cleaning
comb=train.append(test,ignore_index=True)


#removing punctuation,usernames,links,hashtags from tweets
def remove(x):
    x=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",x).split())
    return x
#new column 'clean_tweet' is used to store cleaned tweet
comb['clean_tweet']=comb['tweet'].map(lambda y: remove(y))


#converting into lowercase
comb['clean_tweet']=comb['clean_tweet'].map(lambda y:str(y).lower())

#remove numbers
def remove_digits(x):
    x=''.join([i for i in x if not i.isdigit()])
    return x
comb['clean_tweet']=comb['clean_tweet'].map(lambda y:remove_digits(str(y)))


#remove non-ascii characters
def remove_non_ascii(x):
    x=''.join(i for i in x if ord(i)<128)
    return x
comb['clean_tweet']=comb['clean_tweet'].map(lambda y:remove_non_ascii(y))


#tokenizing tweets
from nltk.tokenize import word_tokenize
def tokenize(x):
    x=word_tokenize(x)
    return x
comb['clean_tweet']=comb['clean_tweet'].map(lambda y: tokenize(y))


#remove stopwords
from nltk.corpus import stopwords
words=set(stopwords.words('english'))
def remove_stopwords(x):
    x=[i for i in x if i not in words]
    return x
comb['clean_tweet']=comb['clean_tweet'].map(lambda y:remove_stopwords(y))



#Lemmatization to return base or dictionary form of a word
from nltk.stem.wordnet import WordNetLemmatizer
l=WordNetLemmatizer()
def lemmatizer(x):
    x=[l.lemmatize(i) for  i in x]
    return x
comb['clean_tweet']=comb['clean_tweet'].map(lambda y:lemmatizer(y))


#remove whitespace characters
def remove_whitespace(x):
    x=[i.replace(' ','') for i in x]
    return x
comb['clean_tweet']=comb['clean_tweet'].map(lambda y:remove_whitespace(y))


#detokenize
def detokenize(x):
    l=' '.join(x)
    x=l
    return x
comb['clean_tweet']=comb['clean_tweet'].map(lambda y:detokenize(y))


#wordcloud to visualize words
'''all_words=' '.join([text for text in comb['clean_tweet']])
from wordcloud import WordCloud
wordcloud=WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()



#words in non-racist tweets
pos_words =' '.join([text for text in comb['clean_tweet'][comb['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(pos_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


#words in racist tweets
neg_words = ' '.join([text for text in comb['clean_tweet'][comb['label'] == 1]])

wordcloud = WordCloud(width=800, height=500,random_state=21, max_font_size=110).generate(neg_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()'''


#feature extraction using tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer=TfidfVectorizer(max_df=0.85,min_df=3,max_features=1000,stop_words='english')

tfidf=tfidf_vectorizer.fit_transform(comb['clean_tweet'])


#separating into train and validation sets

train_tfidf=tfidf[:31962,:]
test_tfidf=tfidf[31962:,:]

from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid=train_test_split(train_tfidf,train['label'],random_state=42,test_size=0.3)


#model building
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

lreg=LogisticRegression()
mnb=MultinomialNB()
per=Perceptron()



#logistic regression model
from IPython.display import display
lreg.fit(x_train,y_train)

y_pred=lreg.predict_proba(x_valid)
y_pred=y_pred[:,1]>=0.3
y_pred=y_pred.astype(np.int)

display(f1_score(y_valid,y_pred))
display(confusion_matrix(y_valid,y_pred))


mnb.fit(x_train,y_train)

y_pred_mnb=mnb.predict(x_valid)

display(f1_score(y_valid,y_pred_mnb))
display(confusion_matrix(y_valid,y_pred_mnb))


#Perceptron
per.fit(x_train,y_train)

y_pred_per=per.predict(x_valid)

display(f1_score(y_valid,y_pred_per))
display(confusion_matrix(y_valid,y_pred_per))



#using MultinomialNB model to predict labels of test data
y_test_pred=mnb.predict(test_tfidf)

test['label']=y_test_pred
submission=test[['id','label']]

#writing to csv file
submission.to_csv("submission.csv",index=False)
