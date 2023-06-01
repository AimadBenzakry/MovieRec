import joblib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

#----------- LOAD MOVIELENS Dataset ml-100k------------#
print("*********LOAD DATA*********")
#loading u.info the number of users, items and ratings in the u data set 
overall_stats = pd.read_csv('ml-100k/u.info', header=None)
print("-Details of users, items and ratings involved in the loaded movielens dataset: ",list(overall_stats[0]))
# print tha table of dataset with the names of columns
column_names1 = ['user id','movie id','rating','timestamp']
dataset = pd.read_csv('ml-100k/u.data', sep='\t',header=None,names=column_names1)
print("-DATASET HEAD :")
print(dataset.head()) 
# print the length of dataset and max number of movies and min number of movies 
print("-Length of dataset , max nb movies, min nb movies :")
len(dataset), max(dataset['movie id']),min(dataset['movie id'])
print(len(dataset), max(dataset['movie id']),min(dataset['movie id']))
#print informations about the items u.item
d = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
column_names2 = d.split(' | ')
print("-U.item , informations about items: ")
print(column_names2)
items_dataset = pd.read_csv('ml-100k/u.item', sep='|',header=None,names=column_names2,encoding='latin-1')
print(items_dataset)
movie_dataset = items_dataset[['movie id','movie title']]
print("-Informations about the items movies id and movie title : ")
print(movie_dataset.head())
#looking at length of original items_dataset and length of unique combination of rows in items_dataset after removing movie id column
print("-Length of original items_dataset ")
print(len(items_dataset.groupby(by=column_names2[1:])),len(items_dataset))
# merging dataset
merged_dataset = pd.merge(dataset, movie_dataset, how='inner', on='movie id')
print("-Merged Datset :")
print(merged_dataset.head())
# example of a rating of a movie by a user
print("-Example of a rating movie of a user :")
print(merged_dataset[(merged_dataset['movie title'] == 'Chasing Amy (1997)') & (merged_dataset['user id'] == 894)])
#print table of rating by user id, movie title , rating
print("-Table of rating by user id, movie title , rating") 
refined_dataset = merged_dataset.groupby(by=['user id','movie title'], as_index=False).agg({"rating":"mean"})
print(refined_dataset.head())

#----------- EXPLORATORY DATA ANALYSIS EDA Dataset ml-100k------------#
#get the counts of each rating from ratings data to ==> Plot the counts of each rating
print("********** EDA ***********")
num_users = len(refined_dataset['user id'].value_counts())
num_items = len(refined_dataset['movie title'].value_counts())
print('-Unique number of users in the dataset: {}'.format(num_users))
print('-Unique number of movies in the dataset: {}'.format(num_items))
rating_count_df = pd.DataFrame(refined_dataset.groupby(['rating']).size(), columns=['count'])
print("table de nombre de ratings")
print(rating_count_df)
#plot count for each rating score in Log Count
rating_count_df.reset_index().rename(columns={'index': 'rating score'}).plot(x='rating', y='count', kind='bar',
    figsize=(12, 8),
    title='Count for Each Rating Score',
    fontsize=12)
plt.xlabel("movie rating score")
plt.ylabel("number of ratings")
plt.show()
# REMARQUE : We can see that number of 1.5, 2.5, 3.5, 4.5 ratings by the users are comparitively negligible.
#number of ratings that are rated = 0.0
total_count = num_items * num_users
zero_count = total_count-refined_dataset.shape[0]
print("-Number of ratings that are rated = 0.0")
print(zero_count)
rating_count_df = rating_count_df.append(
    pd.DataFrame({'count': zero_count}, index=[0.0]),
    verify_integrity=True,
).sort_index()
print(rating_count_df)
# REMARQUE : Number of times no rating was given (forged as 0 in this case) is a lot more than other ratings.
# add log count
rating_count_df['log_count'] = np.log(rating_count_df['count'])
print("Table Logcount : ")
print(rating_count_df)
rating_count_df = rating_count_df.reset_index().rename(columns={'index': 'rating score'})
print(rating_count_df)
#plot count for each rating score in Log Count
rating_count_df.plot(x='rating score', y='log_count', kind='bar', figsize=(12, 8),
    title='Count for Each Rating Score (in Log Scale)',
    logy=True,
    fontsize=12)
plt.xlabel("movie rating score")
plt.ylabel("number of ratings")
plt.show()
# REMARQUE : We have already observed from the before bar plot that ratings 3 and 4 are given in more numbers by the users. Even the above graph suggests the same.
#print rating frequency of all movies
print(refined_dataset.head())
# get rating frequency
movies_count_df = pd.DataFrame(refined_dataset.groupby('movie title').size(), columns=['count'])
print(movies_count_df.head())
#plot the rating frequency of all movies 
movies_count_df_sorted = movies_count_df.sort_values('count', ascending=False).reset_index(drop=True)
plt.figure(figsize=(12, 8))
plt.plot(movies_count_df_sorted.index, movies_count_df_sorted['count'])
plt.title('Rating Frequency of All Movies')
plt.xlabel("movie Id")
plt.ylabel("number of ratings")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#----------------TRAINING KNN MODEL-------------------#
print("#****** TRAIN KNN MODEL*******#")
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
# pivot and create movie-user matrix
movie_to_user_df = refined_dataset.pivot(
     index='movie title',
   columns='user id',
      values='rating').fillna(0)
print("-Movie-User Matrice :")
print(movie_to_user_df.head())
#extracting movie names into a list
# movies_list = list(movie_to_user_df.index)
movies_list = list(refined_dataset['movie title'].unique())
print("-Movie Names List : ")
print(movies_list[:10])
#movie_dict = {movie : index for index, movie in enumerate(movies_list)}
movie_dict = {movie : index for index, movie in enumerate(movies_list)}
print("-Movie Dictionnary : ")
print(movie_dict)
# Update the movie_to_user_df index using the movie_dict
movie_to_user_df = movie_to_user_df.rename(index=movie_dict)
print("-Updated Movie-User Matrix:")
print(movie_to_user_df.head())
# Transform matrix to scipy sparse matrix for better execution
movie_to_user_sparse_df = csr_matrix(movie_to_user_df.values)
print("-Movie-User Sparse Matrix:")
print(movie_to_user_sparse_df)

case_insensitive_movies_list = [i.lower() for i in movies_list]


#fitting knn model : 
#  Define the range of K values to evaluate
k_value = 20
# Initialize lists to store evaluation results
knn_movie_model = NearestNeighbors(n_neighbors=k_value, metric='cosine', algorithm='brute')
squared_errors = []
absolute_errors = []

# Perform cross-validation using KFold
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(movie_to_user_sparse_df):
    train_data = movie_to_user_sparse_df[train_index]
    test_data = movie_to_user_sparse_df[test_index]
    knn_movie_model.fit(train_data)
    
    distances, indices = knn_movie_model.kneighbors(test_data)
        
    for i in range(len(indices)):
        neighbor_ratings = movie_to_user_df.iloc[indices[i][1:]]  # Exclude the movie itself from neighbors
        predicted_rating = neighbor_ratings.mean(axis=0)
        actual_rating = movie_to_user_df.iloc[test_index[i]]
        squared_errors.append((predicted_rating - actual_rating) ** 2)
        absolute_errors.append(abs(predicted_rating - actual_rating))
        
# Load the trained model
joblib.dump(knn_movie_model, 'recommender_model.joblib')

# mse = np.mean(squared_errors)
# mae = np.mean(absolute_errors) 

# # Plot the performance
# plt.bar(['MSE', 'MAE'], [mse, mae])
# plt.xlabel('Metric')
# plt.ylabel('Score')
# plt.title('KNN Model Performance for k=10')
# plt.show()

# ## function to find top n similar movies of the given input movie 
# def get_similar_movies(movie_list, n=10):
#     results = []
#     for movie in movie_list:
#         index = movie_dict[movie]
#         knn_input = np.asarray([movie_to_user_df.values[index]])
#         distances, indices = knn_movie_model.kneighbors(knn_input, n_neighbors=n + 1)
#         print("Top", n, "movies that are very similar to the Movie -", movie, "are:")
#         print(" ")
#         for i in range(1, len(distances[0])):
#             print(movies_list[indices[0][i]])
#             results.append(movies_list[indices[0][i]])
#         print(" ")
#     return results

# # Testing the recommender system:
# movie_list = ['Pocahontas (1995)', 'Toy Story (1995)', 'Love Is All There Is (1996)', 'Beautiful Girls (1996)']
# similar_movies = get_similar_movies(movie_list, 5)
# print("k value : ",k_value)
