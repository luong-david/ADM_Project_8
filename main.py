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

def get_user_information():
    uid = input("Enter your user id (use 0 for test user or -1 for random): ")
    if uid == '0':
        uid = '-OGWTHZng0QNhvc8dhIjyQ'
    elif uid == '-1':
        uid = users[random.randrange(0,len(users))]['user_id']
    return uid

# build user and restaurant lists
user_list = []
for u in users:
    user_list.append(u['user_id'])
rest_list = []
for r in restaurants:
    rest_list.append(r['business_id'])

# Get user information
uid = get_user_information()

# Prompt for user id
while 1:
    if uid in user_list:
        print('Your user id is ' + uid)
        break
    else:
        print('Invalid user id...try again')
        uid = get_user_information()

quick_rec = input("Enter 1 for quick recommendation, 0 to examine all restaurants: ")
while quick_rec != '1' and quick_rec != '0':
    quick_rec = input("I did not understand your input, enter 1 for quick recommendation, 0 to examine all restaurants: ")
top_k = input("How many recs do you want? ")
if int(quick_rec):
    extra_recs = input("May I suggest more recs for you similar to the top-k? (1=yes,0=no) ")
    while extra_recs != '1' and extra_recs != '0':
        extra_recs = input('I did not understand your input, may I suggest more recs for you similar to the top-k? (1=yes,0=no) ')
else:
    extra_recs = 0

# Outputs
write2csv = 1

# Advanced Data Mining Studies
likes,neutral,dislikes,picks,dispicks,extra_picks,user_profile,att_plus,att_minus = content_recommender.recommend(uid,quick_rec,top_k,extra_recs,users,restaurants,reviews,write2csv)
print('You liked: ' + str(likes))
print('You disliked: ' + str(dislikes))
print('Your Top k: ')
print(picks)
print('Your Extra Quick Picks: ')
print(extra_picks)
print('Your Last k: ')
print(dispicks)
print('You prefer these attributes: ')
print(att_plus)
print('You don\'t care for these attributes: ')
print(att_minus)

x2 = cf_item.recommend(users,restaurants,reviews)
x3 = cf_user.recommend(uid,reviews)
x4 = latent_factor_model.recommend(users,restaurants,reviews)
x5 = svd4rec.recommend(users,restaurants,reviews)
x6 = clustering.recommend(users,restaurants,reviews)