
# coding: utf-8

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix


df=pd.read_csv('cleaned_data.csv')

#dividing the data into train and test sets
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df['SentimentText'],df['Sentiment'],random_state=42,test_size=0.2)


#creating classifiers
names = ["Logistic Regression", "Linear SVC","Multinomial NB", 
         "Bernoulli NB", "Ridge Classifier","Perceptron"]
classifiers = [
    LogisticRegression(),
    LinearSVC(),
    MultinomialNB(),
    BernoulliNB(),
    RidgeClassifier(),
    Perceptron()]

zipped_clf = zip(names,classifiers)

print ('Training classifiers...')
print (' ')

#creating pipeline to test all the classifiers
def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    sentiment_fit = pipeline.fit(x_train.values.astype('U'), y_train)
    y_pred = sentiment_fit.predict(x_test.values.astype('U'))
    accuracy = accuracy_score(y_test, y_pred)
    matrix=confusion_matrix(y_test,y_pred)
    
    return accuracy,matrix
    

#extracting features using tf_idf vectorizer
tvec = TfidfVectorizer()
def classifier_comparator(vectorizer=tvec, n_features=10000, stop_words=None, ngram_range=(1, 1), classifier=zipped_clf):
    result = []
    result1=[]
    vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
    for n,c in classifier:
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', c)
        ])
        print (c)
        print (' ')
        clf_accuracy,clf_matrix= accuracy_summary(checker_pipeline, x_train, y_train, x_test, y_test)
        result.append((n,clf_accuracy))
        result1.append((n,clf_matrix))
    return result,result1

trigram_result,trigram_matrix = classifier_comparator(n_features=100000,ngram_range=(1,3))


print ('The accuracy of various models are')
print (' ')
print (trigram_result)
print (' ')
print ('The confusion matrix of models are')
print (trigram_matrix)

