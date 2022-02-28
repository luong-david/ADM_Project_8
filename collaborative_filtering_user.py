import scipy
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


def recommend(userid_input, reviews):
    user_input = str(userid_input)

    # Load the data
    user_ids = []
    business_ids = []
    stars = []

    for x in range(0, len(reviews)):
        user_ids.append(reviews[x]['user_id'])
        business_ids.append((reviews[x]['business_id']))
        stars.append(np.float64((reviews[x]['stars'])))

    if user_input not in user_ids:
        print(user_input, " is not a valid id")
        quit()

    # build utility matrix
    matrix_dict = {"user_id": user_ids, 'business_id': business_ids, 'stars': stars}
    matrix_df = pd.DataFrame(matrix_dict, columns=['user_id', 'business_id', 'stars'])

    pivot_matrix = matrix_df.pivot_table(index=['user_id'], columns=['business_id'], values=['stars']).fillna(0)

    # normalize matrix
    pivot_matrix_norm = pivot_matrix.sub(pivot_matrix.mean(axis=1), axis=0)
    # Get Pearson Correlation
    pivot_matrix_norm_corr = pivot_matrix_norm.T.corr(method='pearson')
    # Remove diagnol 1's in Pearson Correlation
    pivot_matrix_norm_corr.values[[np.arange(pivot_matrix_norm_corr.shape[0])] * 2] = 0

    # KNN
    review_matrix = csr_matrix(pivot_matrix.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(review_matrix)
    distances, indices = model_knn.kneighbors(pivot_matrix.loc[user_input, :].values.reshape(1, -1), n_neighbors=10)
    print("Recommendation for other users similar to user: ", user_input)
    for i in range(1, len(distances.flatten())):
        print('{0}: {1}, with distance of {2}'.format(i, pivot_matrix.index[indices.flatten()[i]],
                                                      distances.flatten()[i]))

    return 0
