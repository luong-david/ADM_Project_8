# -*- coding: utf-8 -*-
"""

Main driver file for data mining studies on Yelp Dataset

Created on Thu May 22 14:2:21 2021

@author: David Luong
"""
#%%
import json
import numpy as np
import scipy as sp
#matplotlib inline
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random
import bisect
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigs
from numpy.linalg import matrix_rank
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
# Import recommenders
import content_recommender
import collaborative_filtering_item as cf_item
import collaborative_filtering_user as cf_user
import latent_factor_model
import svd4rec
import clustering
 
# Open JSON file
restaurant_data = open('yelp_dataset/TX_restaurants.json')
restaurants = json.load(restaurant_data)
print('Total number of restaurants in dataset: ', len(restaurants))

user_data = open('yelp_dataset/TX_users.json')
users = json.load(user_data)
print('Total number of users in dataset: ', len(users))

review_data = open('yelp_dataset/TX_reviews.json')
reviews = json.load(review_data)
print('Total number of reviews in dataset: ', len(reviews))

# Get user profile
uid = input("Enter your user id (use 0 for test user or -1 for random): ")
if uid == '0':
    uid = '-OGWTHZng0QNhvc8dhIjyQ'
elif uid == '-1':
    uid = users[random.randrange(0,len(users))]['user_id']
else:
    print('Invalid user id')
print('Your user id is ' + uid)

# Advanced Data Mining Studies
likes,neutral,dislikes,picks,dispicks,user_profile,att_plus,att_minus = content_recommender.recommend(uid,restaurants,reviews)
print('User likes: ' + str(likes))
print('User dislikes: ' + str(dislikes))
#print('User indifferent ' + str(neutral))
print('Top k: ')
print(picks)
print('Last k: ')
print(dispicks)
print('You prefer: ')
print(att_plus)
print('You don\'t care for: ')
print(att_minus)
x2 = cf_item.recommend(users,restaurants,reviews)
x3 = cf_user.recommend(users,restaurants,reviews)
x4 = latent_factor_model.recommend(users,restaurants,reviews)
x5 = svd4rec.recommend(users,restaurants,reviews)
x6 = clustering.recommend(users,restaurants,reviews)