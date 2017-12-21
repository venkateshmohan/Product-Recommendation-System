# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:04:23 2017

@author: Venkatesh T Mohan
"""
from flask import Flask,render_template,request,jsonify
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:04:23 2017

@author: Venkatesh T Mohan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import pickle

product_data = pd.read_csv('reviews.csv')
del product_data['reviews.userCity']
del product_data['reviews.userProvince']
del product_data['sizes']
del product_data['upc']
del product_data['weight']
del product_data['ean']
product_data.shape
product_data['text length'] = product_data['reviews.text'].apply(len)

product_data_class = product_data[(product_data['reviews.rating'] == 5) | (product_data['reviews.rating'] == 4) | (product_data['reviews.rating'] == 3)|(product_data['reviews.rating'] == 2)|(product_data['reviews.rating'] == 1)]
X = product_data_class['reviews.text']
Y = product_data_class['reviews.rating']
#print(X[0])

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)   
    # Now just remove any stopwords
    return nopunc

#print(text_process(product_data['reviews.text']))
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
# print(len(bow_transformer.vocabulary_))
#review25 = X[14]
# =============================================================================
#print(review25)
#bow25 = bow_transformer.transform([review25])
#print(bow25)
#print(bow25.shape)
#print(bow_transformer.get_feature_names()[98])
#print(bow_transformer.get_feature_names()[6000])  
X = bow_transformer.transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
filename = 'finalized_model.sav'
pickle.dump(nb, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
preds = loaded_model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
#positive_review = product_data['reviews.text'][187]
#print(positive_review)
#positive_review_transformed = bow_transformer.transform([positive_review])
#print(positive_review_transformed)

#print(nb.predict(positive_review_transformed)[0])
copy_product_data = product_data
for index,row in copy_product_data.iterrows():
    if (pd.isnull(row['reviews.rating'])):
        text_of_null = row['reviews.text']
        text_of_null_transformed = bow_transformer.transform([text_of_null])
        rating_prediction = nb.predict(text_of_null_transformed)[0]
        copy_product_data.at[index,'reviews.rating']=rating_prediction
        
final_ratings_mean=copy_product_data.groupby('name').mean()
del final_ratings_mean['reviews.numHelpful']
del final_ratings_mean['text length']
ch=input("Product Recommendation System")
while(ch!='quit'):
 product_1=str(input("Enter first product name: "))
 product_2=str(input("Enter second product name: "))
 average_rating_1=final_ratings_mean.loc[final_ratings_mean.index==product_1,['reviews.rating']]
 print(average_rating_1)
 average_rating_2=final_ratings_mean.loc[final_ratings_mean.index==product_2,['reviews.rating']]
 print(average_rating_2)
 ch=input("continue or quit")
 if ch=='continue':
    product_1=str(input("Enter first product name: "))
    product_2=str(input("Enter second product name: "))
    average_rating_1=final_ratings_mean.loc[final_ratings_mean.index==product_1,['reviews.rating']]
    print(average_rating_1)
    average_rating_2=final_ratings_mean.loc[final_ratings_mean.index==product_2,['reviews.rating']]
    print(average_rating_2)
    ch=input("continue or quit")

# =============================================================================
# for index,row in final_ratings_mean.iterrows():
#     if([i for i in final_ratings_mean.index]==product_1):
#         average_rating=row['reviews.rating']
#         print(average_rating)
# =============================================================================
        



        
        




