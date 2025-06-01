🎬 Movie Recommender System

This is a content-based movie recommender system built using Python, Pandas, Scikit-learn, and Streamlit.
🔍 Description

The application recommends movies similar to a selected movie based on its genres, keywords, cast, crew, and overview. It uses a bag-of-words model to generate movie feature vectors and calculates cosine similarity to identify the most similar movies.
💡 Features

    Loads and merges movie and credit data from the TMDB 5000 dataset

    Extracts and preprocesses textual features such as genre, cast, and director

    Vectorizes movie metadata using CountVectorizer and computes similarity scores

    Interactive web interface using Streamlit for selecting a movie and viewing top 5 recommendations

🛠️ How to Run

    Clone the repository:

git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

    streamlit run movie_recommender_app.py

📁 Dataset

    tmdb_5000_movies.csv

    tmdb_5000_credits.csv

Ensure both CSV files are in the same directory as movie_recommender_app.py.
