# ============================================
# Movie Recommendation System (Task 5)
# Dataset: MovieLens 100k (or ratings.csv from Kaggle)
# ============================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# 1. Load Dataset
ratings = pd.read_csv("ratings.csv")
ratings.dropna()
ratings.head()

# Keep only relevant columns
ratings = ratings[['userId', 'movieId', 'rating']]

print("Dataset Shape:", ratings.shape)
print("Unique Users:", ratings['userId'].nunique())
print("Unique Movies:", ratings['movieId'].nunique())


# 2. Create User-Item Matrix
# Pivot table: rows = users, columns = movies, values = ratings
# Fill missing ratings with 0 (sparse matrix)
user_item_matrix = ratings.pivot_table(index='userId',columns='movieId',values='rating').fillna(0)

print("User-Item Matrix Shape:", user_item_matrix.shape)

# Convert a pivot table to a numpy array
user_item_matrix_values = user_item_matrix.values
user_ids = user_item_matrix.index
movie_ids = user_item_matrix.columns


# 3. Compute User Similarity (Cosine)
# Each row is a user vector (movie ratings)
# Cosine similarity gives similarity between users
user_similarity = cosine_similarity(user_item_matrix_values)
print("User-User Similarity Matrix Shape:", user_similarity.shape)


# 4. User-Based Collaborative Filtering
def recommend_movies_user_based(user_id, k_neighbors=10, top_n=10):
    """
    Recommend movies for a given user using User-Based Collaborative Filtering.
    :param user_id: target userId
    :param k_neighbors: number of similar users to consider
    :param top_n:a number of movies to recommend
    :return:a list of recommended movieIds
    """
    # Map userId to index in matrix
    if user_id not in user_ids:
        return []

    user_index = user_ids.get_loc(user_id)

    # Get similarity scores for this user with all others
    sim_scores = user_similarity[user_index]
    sim_scores[user_index] = 0  # ignore self-similarity

    # Find top-k similar users
    neighbors_idx = np.argsort(sim_scores)[-k_neighbors:]
    neighbors_sims = sim_scores[neighbors_idx]

    # Get their ratings
    neighbors_ratings = user_item_matrix_values[neighbors_idx]

    # Weighted average of neighbor ratings
    weighted_ratings = np.dot(neighbors_sims, neighbors_ratings) / (neighbors_sims.sum() + 1e-9)

    # Exclude movies already rated by the user
    user_ratings = user_item_matrix_values[user_index]
    weighted_ratings[user_ratings > 0] = -np.inf

    # Recommend top-N movies
    recommended_indices = np.argsort(weighted_ratings)[-top_n:][::-1]
    recommended_movie_ids = movie_ids[recommended_indices]
    return recommended_movie_ids



# 5. Matrix Factorization with SVD
# Truncated SVD approximates the user-item matrix in lower dimensions (latent factors)
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(user_item_matrix_values)
item_factors = svd.components_

# Reconstruct predicted ratings matrix
pred_matrix = np.dot(user_factors, item_factors)


def recommend_movies_svd(user_id, top_n=10):
    """
    Recommend movies for a given user using Matrix Factorization (SVD).
    :param user_id: Target userId
    :param top_n: number of movies to recommend
    :return: a list of recommended movieIds
    """
    if user_id not in user_ids:
        return []

    user_index = user_ids.get_loc(user_id)
    preds = pred_matrix[user_index]

    # Exclude already rated movies
    user_ratings = user_item_matrix_values[user_index]
    preds[user_ratings > 0] = -np.inf

    # Get top-N movie indices
    recommended_indices = np.argsort(preds)[-top_n:][::-1]
    recommended_movie_ids = movie_ids[recommended_indices]

    return recommended_movie_ids


# 6. Evaluation: Precision@K
def precision_at_k(recommender_fn, test_data, k=10):
    """
    Compute Precision@K for a recommender system.
    :param recommender_fn: Recommendation function (e.g., recommend_movies_user_based)
    :param test_data: held out test ratings dataframe
    :param k: top-k items to consider
    :return: average precision@k
    """
    precisions = []
    for user in test_data['userId'].unique():
        # Get top-k recommendations
        recs = recommender_fn(user, top_n=k)

        if len(recs) == 0:
            continue

        # Actual movies rated in test set
        actual = set(test_data[test_data['userId'] == user]['movieId'])

        # Hits = intersection
        hits = len(set(recs) & actual)

        precisions.append(hits / k)

    return np.mean(precisions)


# 7. Example Usage
# Pick a random user
sample_user = ratings['userId'].iloc[0]

print("\nRecommendations for User:", sample_user)
print("User-Based CF:", recommend_movies_user_based(sample_user, top_n=10).tolist())
print("SVD-Based:", recommend_movies_svd(sample_user, top_n=10).tolist())
