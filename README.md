# ADM_Project_8
 
Project Title: Recommender System for Yelp Dataset

Team Members: Aaron Choi, Albert Giang, David Luong, Nelson Paz

RECOMMENDERS

•	Content-based Recommender (recommend restaurants to a user similar to previous restaurants rated highly by the user)
    Restaurant profile features: boolean attributes
    
    Build User Profile
    
    Make Predictions (find items closest to user profile via LSH or compare all restaurants if computationally tractable)
    
    Running this model: main.py with runCBrec = 1
    
    
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

•	Latent Factor Model (predict users' reviews for restaurants)

    Find P and Q through matrix factorization

Note: Download dataset from https://www.yelp.com/dataset, un-tar the files, place JSON files into yelp_dataset folder.
