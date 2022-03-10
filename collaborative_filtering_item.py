import numpy as np
import pandas as pd


def recommend(userid_input, restaurantid_input, reviews, restaurants):
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

    # k Nearest Neighbors
    k = 20
    item_top_matrix = pivot_matrix_norm_corr.sort_values(by=[item_input], ascending=False)
    item_top = item_top_matrix[item_input]

    top_businesses = item_top.index.values[1:k+1]
    top_correlations = item_top.values[1:k+1]

    # Top Results
    print("Recommendation for other restaurants similar to restaurant: ", item_input)

    restaurant_name = []
    for x in range(len(restaurants)):
        for y in range(len(top_businesses)):
            if restaurants[x]['business_id'] == top_businesses[y]:
                restaurant_name.append(restaurants[x]['name'])

    item_results_df = pd.DataFrame(data=[top_businesses, restaurant_name, top_correlations]).T
    item_results_df.columns = ["business_id", "Restaurant", "Similarity"]
    print(item_results_df)

    #Predicted Ratings
    user_matrix_dict = {"user_id": user_ids, 'business_id': business_ids, 'stars': stars}
    user_matrix_df = pd.DataFrame(user_matrix_dict, columns=['user_id', 'business_id', 'stars'])
    user_pivot_matrix = user_matrix_df.pivot_table(index=['user_id'], columns=['business_id'], values=['stars'])

    item_pivot_matrix_means = pivot_matrix.mean(axis=1)
    user_pivot_matrix_means = user_pivot_matrix.mean(axis=1)
    item_pivot_matrix_entiremean = np.nanmean(pivot_matrix)

    pred_rating = 0
    sim_numer = []
    sim_denom = []
    mu = item_pivot_matrix_entiremean
    b_x = user_pivot_matrix_means[user_input] - mu
    b_i = item_pivot_matrix_means[item_input] - mu
    b_xi = mu + b_x + b_i

    # Calculate predicted rating for each recommended restaurant
    for x in range(len(top_businesses)):
        for y in range(len(reviews)):

            # checking to see if there are other restaurants reviewed by same user
            # if it is, then calculate predicted rating
            if bool(reviews[y]['business_id'] == top_businesses[x]) & bool(
                    reviews[y]['user_id'] == user_input):
                sim_numer.append(top_correlations[x] * (
                            reviews[y]['stars'] - (mu + b_x + item_pivot_matrix_means[top_businesses[x]] - mu)))
                sim_denom.append(top_correlations[x])
                pred_rating = b_xi + (sum(sim_numer) / sum(sim_denom))

    print("Predicted Rating for Restaurant: ", item_input, "is ", pred_rating)

    return 0
