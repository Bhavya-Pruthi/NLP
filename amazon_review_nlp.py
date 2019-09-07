# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 06:40:00 2019

@author: bhavy
"""
import numpy as np
import pandas as pd
data=pd.read_csv("Amazon_Unlocked_Mobile.csv")
df = data.sample(frac=0.1, random_state=10)
df.dropna(inplace=True)
df=df[df["Rating"]!=3]
df["Positive_Rating"]=np.where(df["Rating"]>3,1,0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                    df['Positive_Rating'], 
                                                    random_state=0)

#Just Using Count Vector
from sklearn.feature_extraction.text import CountVectorizer
# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
len(vect.get_feature_names())
vect_X_train=vect.transform(X_train)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(vect_X_train, y_train)

from sklearn.metrics import roc_auc_score
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

#using tfidf 
#returns a float in sparse matrix 

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=5).fit(X_train)
#length of features is less than that of count vector
len(vect.get_feature_names())

X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

#using n-grams (2 collective words)
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
#has a large no of features
len(vect.get_feature_names())

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))