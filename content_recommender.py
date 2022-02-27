
from cmath import nan
from scipy.sparse import csr_matrix
import numpy as np
import random
from random import randint
from itertools import combinations
import csv

# Python3 Program to check whether a 
# given key already exists in a dictionary.
  
# Function to print boolean
def checkKey(dict, key):
      
    if key in dict:
        if dict[key] == 'True':
            return 1
        else:
            return 0
    else:
        return 0

def band_compare(a,b):
    c = np.array(a)==np.array(b)
    if c is False or len(c) == 0:
        return 0
    else:
        return len(c[c==True])/len(c)

def jaccard_similarity(x,y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity

def build_characteristic_matrix(users,restaurants,reviews):
    mat = csr_matrix((len(users),len(restaurants)),
                  dtype = 'f').toarray()
    user_list = []
    for u in users:
        user_list.append(u['user_id'])
    rest_list = []
    for r in restaurants:
        rest_list.append(r['business_id'])
    for rev in reviews:
        u_idx = user_list.index(rev['user_id'])
        r_idx = rest_list.index(rev['business_id'])
        mat[u_idx][r_idx] = 1
    # remove restaurants without any reviews from users in user_list
    columns = mat.shape[1]
    count = 0
    idx_keep = []
    for i in range(columns):
        if sum(mat[:,i]) != 0:
            idx_keep.append(i)
            count+=1
    mat_filt = csr_matrix((len(users),count),
                dtype = 'f').toarray()
    rest_list_filt = []
    i = 0
    for idx in idx_keep:
        mat_filt[:,i] = mat[:,idx]
        i+=1
        rest_list_filt.append(rest_list[i])
    return mat_filt, idx_keep

def build_signature_matrix(mat,nBands,nRows):
    # Set random seed for repeatability
    random.seed(10)

    # Partition Signature Matrix into nBands bands and nRows rows      
    nF = nBands*nRows

    # Create hash functions
    m = 1 #len(users)
    p = 809 # next prime after len(users) obtained from http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
    a = [randint(0, 100) for p in range(0, nF)]
    b = [randint(0, 100) for p in range(0, nF)]

    # Create signature matrix via hashing (faster way)

    # Initialize to large values
    mat_sig = np.ones((nF,mat.shape[1]))*99999

    # Scan rows of characteristic matrix
    nR=0
    for row in mat:
        np_row = np.array(row)
        item_index = np.where(np_row==1)
        for nC in item_index[0]:
            for i in range(nF):
                h = ((a[i]*nR+b[i])%p)%m
                if h < mat_sig[i][nC]:
                    mat_sig[i][nC] = h
        nR+=1
    return mat_sig

def collect_candidate_pairs(mat,mat_sig,nBands,nRows,s):
    final_result = []

    for b in range(nBands):
        mydict = {}
        result = []
        print('Finding candidate pairs in band ' + str(b+1) + ' of ' + str(nBands))
        start = b*nRows+1
        end = nRows*(b+1)+1
        for j in range(mat_sig.shape[1]):
            mydict.setdefault(hash(tuple(mat_sig[start:end,j])), []).append(j)

        for key in mydict.keys():
            cp_idx = mydict[key]
            if len(cp_idx) > 1:
                if len(cp_idx) > 2:
                    comb = combinations(cp_idx,2)
                    for c in list(comb):
                        if band_compare(mat_sig[start:end,c[0]],mat_sig[start:end,c[1]]) == 1:
                            sim_score = jaccard_similarity(mat[:,c[0]],mat[:,c[1]])
                            if sim_score >= s:
                                item = [[c[0],c[1]],sim_score]
                                if item not in final_result:
                                    result.append(item)
                else:
                    if band_compare(mat_sig[start:end,cp_idx[0]],mat[start:end,cp_idx[1]]) == 1:
                        sim_score = jaccard_similarity(mat[:,cp_idx[0]],mat[:,cp_idx[1]])
                        if sim_score >= s:
                            item = [cp_idx,sim_score]
                            if item not in final_result:
                                result.append(item)
        final_result = final_result + result
    
    print('Found ' + str(len(final_result)) + ' candidate pairs above or equal to threshold ' + str(s))
    return final_result

def LSH(users,restaurants,reviews):
    # Parameters
    s = 0.25 # similarity threshold
    nBands = 2
    nRows = 2

    mat,map = build_characteristic_matrix(users,restaurants,reviews)
    mat_sig = build_signature_matrix(mat,nBands,nRows)
    final_result = collect_candidate_pairs(mat,mat_sig,nBands,nRows,s)

    # get buckets of restaurants with original restaurant indices using map
    buckets = {}
    for cp in final_result:
        if map[cp[0][0]] in buckets.keys():
            buckets[map[cp[0][0]]].append(map[cp[0][1]])
        else:
            buckets[map[cp[0][0]]] = [map[cp[0][1]]]

    return buckets

def recommend(uid,quick_rec,top_k,extra_recs,users,restaurants,reviews,write2csv):

    # ensure correct types
    k = int(top_k) # top-k recs
    quick_rec = int(quick_rec)

    # remove restaurants without any attributes
    for i,restaurant in enumerate(restaurants):
        if restaurant['attributes'] == None: 
            print("[Feature Selection] Removing " + restaurant['name'] + ": no attributes")
            del restaurants[i]
    
    # get features (boolean)
    att_list = set()
    for restaurant in restaurants:
        att = [k for k,v in restaurant['attributes'].items() if v == "True" or v == "False"]
        for item in att:
            att_list.add(item)
 
    # build item profiles
    nFeatures = len(att_list)
    nRestaurants = len(restaurants)
    mat = csr_matrix((nRestaurants,nFeatures),
                  dtype = 'f').toarray()
    mat_norm = csr_matrix((nRestaurants,nFeatures),
                  dtype = 'f').toarray()
    for nR, restaurant in enumerate(restaurants):
        for idx, att in enumerate(att_list):
            mat[nR][idx] = checkKey(restaurants[nR]['attributes'],att)

    # normalize item profiles
    for i,row in enumerate(mat):
        if sum(row) == 0:
            continue
        mat_norm[i,:] = mat[i,:]/float(np.sqrt(sum(row)))

    # build user profile from item profile
    # 1. Using review data, find restaurants each user has rated (4-5 stars = +1, 1-3 stars = -1, else 0)
    bid = []
    bstars = []
    user_rated = []
    for rev in reviews:
        if rev['user_id'] == uid:
            bid.append(rev['business_id'])
            bstars.append(float(rev['stars']))
    for nR, restaurant in enumerate(restaurants):
        if restaurant['business_id'] in bid:
            idx = bid.index(restaurant['business_id'])
            if bstars[idx] > 3:
                user_rated.append(1.0)
            elif bstars[idx] <= 3:
                user_rated.append(-1.0)
            else:
                user_rated.append(0.0)
        else:
            user_rated.append(0.0)
    user = np.array(user_rated)
    # generate user profile from user ratings
    user_profile = np.dot(user,mat_norm)

    # Find similar restaurants to user profile (LSH)
    # 1. Create signature matrix with user profile and item profile
    # 2. Compute similarity between candidate pairs in the bucket containing the user profile
    # 3. Recommend top-k restaurants
    # Note: since nRestaurants may be small, allow the option to compute similarity between user profile and all restaurants.

    if quick_rec: # perform LSH to obtain candidate pairs
        buckets = LSH(users,restaurants,reviews)
        cand_pairs = buckets.keys()
    else: # all restaurants are candidate pairs with user profile
        cand_pairs = range(nRestaurants)

    sim = []
    for nR in cand_pairs:
        sim.append((nR,np.dot(user_profile,mat_norm[nR,:])/np.linalg.norm(user_profile,ord=2)/np.linalg.norm(mat_norm[nR,:],ord=2)))
    sim.sort(key=lambda x:x[1],reverse=True)

    # recommend top-k and bottom-k restaurants (and restaurants user has previously rated...k of each rating)
    picks = []
    dispicks = []
    likes = []
    dislikes = []
    neutral = []
    for i,item in enumerate(sim):
        if i < k:
            picks.append([restaurants[item[0]]['name'] + ' on ' + restaurants[item[0]]['address'],item[1]])
        if i > len(cand_pairs)-(k+1):
            dispicks.append([restaurants[item[0]]['name'] + ' on ' + restaurants[item[0]]['address'],item[1]])
    for i,item in enumerate(user):
        if item == 1.0:
            likes.append(restaurants[i]['name'])
        elif item == 0.0:
            neutral.append(restaurants[i]['name'])
        else:
            dislikes.append(restaurants[i]['name'])

    # get user's attributes
    att_plus = []
    att_minus = []
    for i,val in enumerate(user_profile):
        if val > 0:
            att_plus.append(list(att_list)[i])
        else:
            att_minus.append(list(att_list)[i])

    extra_picks = []
    if extra_recs:
        for kk in range(min(k,len(cand_pairs))):
            if sim[kk][0] in cand_pairs:
                extra_picks.append([restaurants[buckets[sim[kk][0]][0]]['name'] + ' on ' + restaurants[buckets[sim[kk][0]][0]]['address'],sim[kk][1]])
    # remove duplicates
    temp = []
    for elem in extra_picks:
        if elem not in temp:
            temp.append(elem)
    extra_picks = temp

    if write2csv:
    # writing to csv file
        if quick_rec:
            tag = 'quick'
        else:
            tag = 'full'
        outvar = [picks,dispicks]
        outname = ['picks','dispicks']
        # field names
        fields = ['Restaurant','Similarity Score']
        for i,out in enumerate(outvar):
            output_file_path = 'content_based_recommender_' + str(outname[i]) + '_' + tag + '.csv'
            with open(output_file_path,'w', encoding='UTF8', newline='') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
                # writing the fields
                csvwriter.writerow(fields)
                # writing the data rows
                csvwriter.writerows(out)

    return likes, neutral, dislikes, picks, dispicks, extra_picks, user_profile, att_plus, att_minus