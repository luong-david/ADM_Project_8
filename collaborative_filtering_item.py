import numpy as np
import pandas as pd


def recommend(userid_input, restaurantid_input,reviews):
    user_input = str(userid_input)
    item_input = str(restaurantid_input)

    # Load the data
    user_ids = []
    business_ids = []
    stars = []

    for x in range(0, len(reviews)):
        user_ids.append(reviews[x]['user_id'])
        business_ids.append((reviews[x]['business_id']))
        stars.append(np.float64((reviews[x]['stars'])))

    if item_input not in business_ids:
        print(item_input, " is not a valid id")
        exit()

    if user_input not in user_ids:
        print(user_input, " is not a valid id")
        exit()

    # build utility matrix
    matrix_dict = {"user_id": user_ids, 'business_id': business_ids, 'stars': stars}
    matrix_df = pd.DataFrame(matrix_dict, columns=['user_id', 'business_id', 'stars'])

    pivot_matrix = matrix_df.pivot_table(index=['business_id'], columns=['user_id'], values=['stars'])

    # normalize matrix
    pivot_matrix_ = pivot_matrix.sub(pivot_matrix.mean(axis=1), axis=0)
    pivot_matrix_norm = pivot_matrix_.replace(np.NaN, 0)

    # Get Pearson Correlation
    pivot_matrix_norm_corr = pivot_matrix_norm.T.corr(method='pearson')
    # Remove diagonal 1's in Pearson Correlation
    pivot_matrix_norm_corr.values[[np.arange(pivot_matrix_norm_corr.shape[0])] * 2] = 0

    # k Nearest Neighbors
    k = 5
    item_top_matrix = pivot_matrix_norm_corr.sort_values(by=[item_input], ascending=False)
    item_top = item_top_matrix[item_input]

    top_businesses = item_top.index.values[0:k]
    top_correlations = item_top.values[0:k]

    # Top Results
    print("Recommendation for other restaurants similar to restaurant: ", item_input)

    for i in range(len(top_businesses)):
        print('{0}: {1}, with similarity of {2}'.format(i, top_businesses[i], top_correlations[i]))



    item_pivot_matrix_means = pivot_matrix.mean(axis=1)
    user_pivot_matrix_means = pivot_matrix.mean(axis=1)
    item_pivot_matrix_entiremean = np.nanmean(pivot_matrix)

    pred_rating = 0
    sim_numer = []
    sim_denom = []
    mu = item_pivot_matrix_entiremean
    b_x = user_pivot_matrix_means[user_input] - mu
    b_i = item_pivot_matrix_means[item_input] - mu
    b_xi = mu + b_x + b_i

    # Calculate predicted rating for each recommended restaurant
    for x in range(len(item_top_businesses)):
        for y in range(len(reviews)):

            # checking to see if there are other restaurants reviewed by same user
            # if it is, then calculate predicted rating
            if bool(reviews[y]['business_id'] == item_top_businesses[x]) & bool(
                    reviews[y]['user_id'] == user_input):
                sim_numer.append(item_top_correlations[x] * (
                            reviews[y]['stars'] - (mu + b_x + item_pivot_matrix_means[item_top_businesses[x]] - mu)))
                sim_denom.append(item_top_correlations[x])
                pred_rating = b_xi + (sum(sim_numer) / sum(sim_denom))

    print("Predicted Rating for Restaurant: ", item_input, "is ", pred_rating)


    return 0
