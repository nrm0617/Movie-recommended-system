# movie_recommender_app.py

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')
    return movies

movies = load_data()

# Utility functions
def convert(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

def collapse(lst):
    return ' '.join(lst)

# Preprocessing
def preprocess(movies):
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3] if isinstance(x, str) else [])
    movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if isinstance(x, str) else [])

    movies['tags'] = movies['overview'].fillna('') + ' ' + \
                     movies['genres'].apply(collapse) + ' ' + \
                     movies['keywords'].apply(collapse) + ' ' + \
                     movies['cast'].apply(collapse) + ' ' + \
                     movies['crew'].apply(collapse)

    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower() if isinstance(x, str) else '')
    return new_df

new_df = preprocess(movies)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    if movie not in new_df['title'].values:
        return ["Movie not found in dataset."]
    idx = new_df[new_df['title'] == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]

# Streamlit UI
st.title("🎬 Movie Recommender System")
selected_movie = st.selectbox("Choose a movie to get recommendations:", new_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Top 5 Recommended Movies:")
    for rec in recommendations:
        st.write(f"🎥 {rec}")

