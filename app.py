import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("Movie Recommendation System 🎬")

@st.cache_resource
def load_data():
    if not os.path.exists('movies_list.pkl') or not os.path.exists('similarity.pkl'):
        return None, None
    movies = joblib.load('movies_list.pkl')
    similarity = joblib.load('similarity.pkl')
    return movies, similarity

movies, similarity = load_data()

if movies is None or similarity is None:
    st.warning("Model artifacts not found! Please run `python train.py` to train the model first.")
else:
    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )
    
    if st.button('Show Recommendations'):
        try:
            movie_idx = movies[movies['title'] == selected_movie].index[0]
            distances = similarity[movie_idx]
            
            movies_list_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
            
            st.subheader(f"Because you watched {selected_movie}, we recommend:")
            
            for i, idx in enumerate(movies_list_indices):
                st.write(f"**{i+1}. {movies.iloc[idx[0]].title}**")
                
        except Exception as e:
            st.error(f"Error finding recommendations: {e}")
