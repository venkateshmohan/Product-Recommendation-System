# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 23:29:29 2017

@author: Aishwarya
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import final_ratings_mean

#storing the dataset into a pandas dataframe "product_data"
product_data = pd.read_csv('reviews.csv')

#preprocessing of the data: deleting the extra columns which we are not considering for our algorithm
del product_data['reviews.userCity'] 
del product_data['reviews.userProvince']
del product_data['sizes']
del product_data['upc']
del product_data['weight']
del product_data['ean']
product_data.shape
product_data.head()
product_data.info()
product_data['text length'] = product_data['reviews.text'].apply(len)

#Plotting a FacetGrid 
g = sns.FacetGrid(data=product_data, col='reviews.rating')
g.map(plt.hist, 'text length', bins=50, color='#28546E')
plt.savefig('stars_textlen', dpi=800)

#Plotting a Box plot to count the length of review texts for each of the star ratings
sns.boxplot(x='reviews.rating', y='text length', data=product_data,
palette=sns.cubehelix_palette(9, start=.4, rot=-.70, reverse=True, light=0.85, dark=0.25))
plt.savefig('stars_textlen_box', dpi=200)

#Plotting the count of number of each star rating
sns.countplot(x='reviews.rating', data=product_data,
palette=sns.cubehelix_palette(9, start=.4, rot=-.70, reverse=True, light=0.85, dark=0.25))


#Calculating the average rating of each product which is mainly the average on the ratings given by the user
previous_mean_of_ratings = product_data.groupby('name').mean()
#deleting two more columns
del previous_mean_of_ratings['reviews.numHelpful']
del previous_mean_of_ratings['text length']
