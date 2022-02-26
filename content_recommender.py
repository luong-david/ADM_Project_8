
from cmath import nan
from scipy.sparse import csr_matrix
import numpy as np

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

def recommend(uid,restaurants,reviews):

    k = 3

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

    # find similar restaurants to user profile (LSH)
    # 1. Create signature matrix with user profile and item profile
    # 2. Compute similarity between candidate pairs in the bucket containing the user profile
    # 3. Recommend top-k restaurants

    # For now since nRestaurants is small, compute similarity between user profile and all restaurants.
    sim = []
    for i,nR in enumerate(range(nRestaurants)):
        sim.append((i,np.dot(user_profile,mat_norm[nR,:])/np.linalg.norm(user_profile,ord=2)/np.linalg.norm(mat_norm[nR,:],ord=2)))
    sim.sort(key=lambda x:x[1],reverse=True)

    # recommend top-k and bottom-k restaurants (and restaurants user has previously rated...k of each rating)
    picks = []
    dispicks = []
    likes = []
    dislikes = []
    neutral = []
    for i,item in enumerate(sim):
        if i < k:
            picks.append(restaurants[item[0]]['name'])
        if i > nRestaurants-(k+1):
            dispicks.append(restaurants[item[0]]['name'])
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

    return likes[0:k],neutral[0:k],dislikes[0:k],picks,dispicks, user_profile, att_plus, att_minus