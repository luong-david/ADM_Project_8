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

# build user and restaurant lists
user_list = []
for u in users:
    user_list.append(u['user_id'])
rest_list = []
for r in restaurants:
    rest_list.append(r['business_id'])

# flags
runCBrec = 1

# Relevance/Metrics Criteria
minStars = 4.0
minRev = 100
k = 10

# Advanced Data Mining Studies

if runCBrec:
    print('====================START CONTENT-BASED RECOMMENDER================================')
    uid, quick_rec, top_N, extra_recs, compute_metrics, write2csv, buckets, cand_pairs = content_recommender.prompt(user_list,users,restaurants,reviews,k)
    AP,AR, likes,neutral,dislikes,picks,extra_picks,user_profile,att_plus,att_minus = content_recommender.recommend(uid,quick_rec,top_N,extra_recs,users,restaurants,reviews,minStars,minRev,k,buckets,cand_pairs,write2csv)
    print('You liked: ' + str(likes))
    print('You disliked: ' + str(dislikes))
    print('Your Top N: ')
    print(picks)
    print('Your Extra Quick Picks: ')
    print(extra_picks)
    print('You prefer these attributes: ')
    print(att_plus)
    print('You don\'t care for these attributes: ')
    print(att_minus)
    print('Average Precision @  ', str(k), ' : ', AP)
    print('Average Recall @  ', str(k), ' : ', AR)
    print('Average F1 Score @ ', str(k), ' : ', 2*AP*AR/(AP+AR))
    if compute_metrics:
        AP_list = []
        AR_list = []
        for i,user in enumerate(user_list):
            AP,AR,likes,neutral,dislikes,picks,extra_picks,user_profile,att_plus,att_minus = content_recommender.recommend(user,quick_rec,top_N,extra_recs,users,restaurants,reviews,minStars,minRev,k,buckets,cand_pairs,write2csv)
            AP_list.append(AP)
            AR_list.append(AR)
            print('User ' + str(i+1) + '/' + str(len(user_list)) + ': Average Precision/Recall @ ' + str(k) + ' = ' + str(AP) + '/' + str(AR))
        print('For ' + str(i+1) + ' out of ' + str(len(user_list)) + ' users: ')

        # Compute metrics
        MAP = sum(AP_list)/len(AP_list)
        MAR = sum(AR_list)/len(AR_list)
        print('MAP = ', MAP)
        print('MAR = ', MAR)
        print('MAF1 = ', 2*MAP*MAR/(MAP+MAR))
    print('====================END CONTENT-BASED RECOMMENDER==================================')
x2_business_id = "-dMuB2gJ2z3wdatazYNu4g"
x2 = cf_item.recommend(x2_business_id,reviews) #Example
x3 = cf_user.recommend(uid,reviews)
x4 = latent_factor_model.recommend(users,restaurants,reviews)
x5 = svd4rec.recommend(users,restaurants,reviews)
x6 = clustering.recommend(users,restaurants,reviews)