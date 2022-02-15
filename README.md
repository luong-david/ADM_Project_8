# ADM_Project_8
 
Project Title: Recommender System for Yelp Dataset

Team Members: Aaron Choi, Albert Giang, David Luong, Nelson Paz

Project idea, including a clear description of the problem and your approach to solve it
•	Content-based Recommender (recommend restaurants to a user similar to previous restaurants rated highly by the user)
    Restaurant profile features: set of friends, set of important words in reviews (use TF-IDF)
    Build User Profile
    Make Predictions (find items closest to user profile via LSH)
•	User-User Collaborative Filtering (recommend new restaurants that were also liked by like-minded reviewers)
    Generate utility matrix (rows: reviewers, columns: restaurants, values: stars)
    Normalize ratings for each reviewer
    Find Pearson correlation between reveiwers
    Use KNN/LSH to find k similar reviewers
    Estimate the rating for a given restaurant based on ratings of k similar reviewers that rated the restaurant (use common practices to model local/global effects)
    Evaluation metrics (MAE, MSE, RMSE, Precision/Recall, F1, ROC, Diversity, Coverage, Serendipity, Novelty, Relevancy, Top N Accuracy)
•	Item-Item Collaborative Filtering (recommend new restaurants that were also liked by the same reviewer)
    Generate the utility matrix (rows: restaurants, columns: reviewers)
    Normalize ratings for each restaurant
    Find Pearson correlation between restaurants
    Use KNN/LSH to find k similar restaurants 
    Estimate the rating for a given restaurant based on ratings of similar restaurants rated by the same reviewer (use common practices to model local/global effects)
    Evaluation metrics (MAE, MSE, RMSE, Precision/Recall, F1, ROC, Diversity, Coverage, Serendipity, Novelty, Relevancy, Top N Accuracy)
•	Hybrid Methods for CF
    Combine multiple recommenders and combine predictions (linear model)
    Content-based + CF
        Item profiles for new restaurants
        Demographics to handle new reviewer problem
•	Latent Factor Model (predict users' reviews for restaurants)
    Find P and Q through matrix factorization
    Extend to include biases and interactions
•	SVD for Recommender Systems
    Generate utility matrix (rows: reviewers, columns: restaurants, values: stars)
    Perform SVD using CUR
    Select k and perform DR, obtain the sliced V
    Define a similarity function e.g. cosine and find the most similar restaurant to a given query
    Recommend the top-n reviewers and identify the restaurant "concept" i.e. type
•	Clustering (understand the structure of reviewers/restaurants)
    Apply BFR and CURE

Note: Download dataset from https://www.yelp.com/dataset, un-tar the files, place JSON files into yelp_dataset folder.
