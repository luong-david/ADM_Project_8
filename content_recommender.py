
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

def recommend(users,restaurants,reviews):

    k = 10

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
    for nR, restaurant in enumerate(restaurants):
        for idx, att in enumerate(att_list):
            mat[nR][idx] = checkKey(restaurants[nR]['attributes'],att)

    # normalize item profiles
    for i,row in enumerate(mat):
        if sum(row) == 0:
            continue
        mat[i,:]/=float(np.sqrt(sum(row)))

    # build user profile from item profile (use random for now)
    # 1. Using review data, find restaurants each user has rated (4-5 stars = +1, 1-3 stars = -1, else 0)
    user = np.random.choice([-1.0,0.0,1.0], size=(1,nRestaurants))
    user_profile = np.dot(user[0],mat)

    # find similar restaurants to user profile (LSH)
    # 1. LSH to find similar restaurants and put into buckets
    # 2. Compute similarity between buckets (choose random restaurant) and user profile
    # 3. Using the most similar bucket, compute similarity for every restaurant in bucket with user profile
    # 4. Recommend top-k restaurants

    # For now since nRestaurants is small, compute similarity between all restaurants.
    sim = []
    for i,nR in enumerate(range(nRestaurants)):
        sim.append((i,np.dot(user_profile,mat[nR,:])/np.linalg.norm(user_profile,ord=2)/np.linalg.norm(mat[nR,:],ord=2)))
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
        if i > nRestaurants-k:
            dispicks.append(restaurants[item[0]]['name'])
    for i,item in enumerate(user[0]):
        if item == 1.0:
            likes.append(restaurants[i]['name'])
        elif item == 0.0:
            neutral.append(restaurants[i]['name'])
        else:
            dislikes.append(restaurants[i]['name'])

    return likes[0:k+1],neutral[0:k+1],dislikes[0:k+1],picks,dispicks, user_profile, att_list