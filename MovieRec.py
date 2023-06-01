import streamlit as st
from streamlit_searchbar import streamlit_searchbar
import random
import requests
import urllib
from PIL import Image
import io


TMDB_API_KEY= "190d8f22637b611332cb63a4a9d33ec1"
def fetch_poster(title):
    url='https://api.themoviedb.org/3/search/movie?api_key={}&query={}'.format(TMDB_API_KEY,title)
    data = requests.get(url) 
    data = data.json()
    poster_path = data['results'][0]['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path



# value = streamlit_searchbar()

##################################
import numpy as np
import joblib
import pandas as pd

model = joblib.load('recommender_model.joblib')

column_names1 = ['user id','movie id','rating','timestamp']
dataset = pd.read_csv('ml-100k/u.data', sep='\t',header=None,names=column_names1)
d = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
column_names2 = d.split(' | ')
items_dataset = pd.read_csv('ml-100k/u.item', sep='|',header=None,names=column_names2,encoding='latin-1')
movie_dataset = items_dataset[['movie id','movie title']]
merged_dataset = pd.merge(dataset, movie_dataset, how='inner', on='movie id')
refined_dataset = merged_dataset.groupby(by=['user id','movie title'], as_index=False).agg({"rating":"mean"})
movies_list = list(refined_dataset['movie title'].unique())
movie_dict = {movie : index for index, movie in enumerate(movies_list)}
movie_to_user_df = refined_dataset.pivot(
     index='movie title',
   columns='user id',
      values='rating').fillna(0)

def get_similar_movies(movie_list, n=10):
    results = []
    for movie in movie_list:
        index = movie_dict[movie]
        knn_input = np.asarray([movie_to_user_df.values[index]])
        distances, indices = model.kneighbors(knn_input, n_neighbors=n + 1)
        print("Top", n, "movies that are very similar to the Movie -", movie, "are:")
        print(" ")
        for i in range(1, len(distances[0])):
            print(movies_list[indices[0][i]])
            results.append(movies_list[indices[0][i]])
        print(" ")
    return results
####################################################################

def main():
    st.title('MovieRec')
    
    def about():
        st.write('Welcome to MovieRec, a movie recommendation engine that uses collaborative filtering to recommend movies based on your selection.')
        st.write('To get started, type in the name of a movie in the search bar below or select a movie from the dropdown menu.')
        st.write('Then click on the "Show Recommendation" button to get your movie recommendations.')
        st.write('Note: The recommendations are based on the MovieLens 100K dataset, which contains 100,000 ratings from 1000 users on 1700 movies. The dataset is available at https://grouplens.org/datasets/movielens/100k/')
        st.write('Enjoy!')
    def home():
        selected_movie_name = st.selectbox(
            "Type or select a movie from the dropdown",
            movies_list,on_change=None
            )
        list___= [selected_movie_name]
   
        button = st.button('Show Recommendation' )
    
        if button:
            recommendations = get_similar_movies(list___)
            st.text("Movie selected: " + selected_movie_name + "")
            if fetch_poster(selected_movie_name.split("(")[0]) is not None:
                poster_path= fetch_poster(selected_movie_name.split("(")[0])
                image_data = urllib.request.urlopen(poster_path).read()
                image = Image.open(io.BytesIO(image_data))
                resized_image = image.resize((300, 300))  # Adjust the dimensions as needed
                st.image(resized_image)
            # st.image(fetch_poster(selected_movie_name.split("(")[0]),width=300)
            st.subheader("Movies Similar to your selection ...")
            for i in recommendations:
                st.write(i)
                if(fetch_poster(i.split("(")[0]) == None):
                    st.write("No poster available")
                    continue
                # Retrieve the poster URL for the current movie
                poster_path= fetch_poster(i.split("(")[0])
                # poster_url = "https://image.tmdb.org/t/p/w500/{}".format(poster_path)
                image_data = urllib.request.urlopen(poster_path).read()
                image = Image.open(io.BytesIO(image_data))
                resized_image = image.resize((300, 300))  # Adjust the dimensions as needed
                st.image(resized_image)
                # Display the poster image
                # st.image(image, use_column_width=True)
                
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        home()
    elif choice == "About":
        about()

    

    

        
if __name__ == '__main__':
    main()
# st.write(fetch_poster('the avengers'))
# poster_path= fetch_poster('avengers')


# poster_url = "https://image.tmdb.org/t/p/w500/{}".format(poster_path)
# image_data = urllib.request.urlopen(poster_url).read()
# image = Image.open(io.BytesIO(image_data))
# resized_image = image.resize((200, 200))  # Adjust the dimensions as needed

# st.image(resized_image)



# movie_list = ['Pocahontas (1995)', 'Toy Story (1995)', 'Love Is All There Is (1996)', 'Beautiful Girls (1996)']
# similar_movies = get_similar_movies(movie_list, 5)
# print(similar_movies)



# st.subheader("Movies You May Like ...")

# # st.write(similar_movies)


# def generate_random_rec():
#     return random.sample(movies_list,20)


# liste = generate_random_rec()

# liked_items = st.session_state.get('liked_items', [])

# scroll_container = st.container()

# scroll_index = st.session_state.get('scroll_index', 0)

# def add_liked(i):
#      liked_items.append(i)

# with scroll_container.container():
#         prev_button, next_button = st.columns([1, 1])

#         if prev_button.button('Prev') and scroll_index > 0:
#             scroll_index -= 1
#         if next_button.button('Next') and scroll_index < len(liste) - 3:
#             scroll_index += 1

#         items = liste[scroll_index:scroll_index + 3]
#         # for item in items:
#         #     st.write(item)

#         columns_ = st.columns(len(items))

#         for i, column in enumerate(columns_):
#                 column.write(items[i])
#                 column.button('Like',items[i], on_click= add_liked(items[i]))




# st.session_state['scroll_index'] = scroll_index

# st.session_state['liked_items'] = liked_items

# st.write('Liked Items:', liked_items)






