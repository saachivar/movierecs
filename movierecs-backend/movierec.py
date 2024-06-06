import sqlite3 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


conn = sqlite3.connect('movielens.db')

# Load data into DataFrames
movies = pd.read_sql_query("SELECT * FROM movies", conn)
ratings = pd.read_sql_query("SELECT * FROM ratings", conn)
tags = pd.read_sql_query("SELECT * FROM tags", conn)
genome_scores = pd.read_sql_query("SELECT * FROM genome_scores", conn)
genome_tags = pd.read_sql_query("SELECT * FROM genome_tags", conn)

# Create a user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Convert to DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_movies(user_id, num_recommendations=5):
    # Get user's similarity scores
    user_scores = user_similarity_df[user_id]
    
    # Sort users by similarity scores
    similar_users = user_scores.sort_values(ascending=False).index[1:]
    
    # Get the movies rated by similar users
    similar_users_ratings = user_item_matrix.loc[similar_users]
    
    # Calculate the weighted average of ratings for each movie
    recommended_movies = similar_users_ratings.apply(lambda row: np.dot(row, user_scores[similar_users]) / sum(user_scores[similar_users]), axis=0)
    
    # Remove movies already rated by the user
    user_rated_movies = user_item_matrix.loc[user_id]
    recommended_movies = recommended_movies[user_rated_movies == 0]
    
    # Sort and get top N recommendations
    top_recommendations = recommended_movies.sort_values(ascending=False).head(num_recommendations)
    
    return movies[movies['movieId'].isin(top_recommendations.index)]

# Example: Recommend movies for user with ID 1
recommendations = recommend_movies(1)
print(recommendations)
