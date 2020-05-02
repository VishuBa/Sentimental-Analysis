
# coding: utf-8



import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize

#loadin dataset into dataframe
df=pd.read_csv('train.csv',encoding='latin-1')
print ('Loaded Dataset')

#creating corpus
corpus1=df.head(100000)
corpus=corpus1.append(df.tail(100000))

# removig unnecessary columns in corpus
corpus.drop(['ItemID','Date','SentimentSource','Blank'],axis=1,inplace=True)

#converting positive sentiment which is labelled as 4 to 1
convert=lambda x: 1 if  x>3 else 0
corpus['Sentiment'].apply(convert)


print ('Pre-processing the data.....')
#removing punctuation,usernames,links,hashtags from tweets
def remove(x):
    x=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",x).split())
    return x
corpus['SentimentText']=corpus['SentimentText'].map(lambda y: remove(y))

#converting into lowercase
corpus['SentimentText']=corpus['SentimentText'].map(lambda y:str(y).lower())

#remove numbers
def remove_digits(x):
    x=''.join([i for i in x if not i.isdigit()])
    return x
corpus['SentimentText']=corpus['SentimentText'].map(lambda y:remove_digits(str(y)))

#remove non-ascii characters
def remove_non_ascii(x):
    x=''.join(i for i in x if ord(i)<128)
    return x
corpus['SentimentText']=corpus['SentimentText'].map(lambda y:remove_non_ascii(y))

#tokenizing tweets
def tokenize(x):
    x=word_tokenize(x)
    return x
corpus['SentimentText']=corpus['SentimentText'].map(lambda y: tokenize(y))

#remove stopwords
from nltk.corpus import stopwords
words=set(stopwords.words('english'))
def remove_stopwords(x):
    x=[i for i in x if i not in words]
    return x
corpus['SentimentText']=corpus['SentimentText'].map(lambda y:remove_stopwords(y))


'''#stemming 
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer('english')
def stem(x):
    x=[stemmer.stem(i) for i in x]
    return x
corpus['tweet']=corpus['tweet'].map(lambda y:stem(y))'''


#Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
l=WordNetLemmatizer()
def lemmatizer(x):
    x=[l.lemmatize(i) for  i in x]
    return x
corpus['SentimentText']=corpus['SentimentText'].map(lambda y:lemmatizer(y))

#remove whitespace characters
def remove_whitespace(x):
    x=[i.replace(' ','') for i in x]
    return x
corpus['SentimentText']=corpus['SentimentText'].map(lambda y:remove_whitespace(y))

#detokenize
def detokenize(x):
    l=' '.join(x)
    x=l
    return x
corpus['SentimentText']=corpus['SentimentText'].map(lambda y:detokenize(y))


'''from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud=WordCloud(background_color='black',height=1000,width=1200).generate(''.join(corpus['tweet']))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()'''

#saving pre-processed data to a csv file
corpus.to_csv('cleaned_data.csv',index=False)
print ('cleaned data saved to a file cleaned.csv')
