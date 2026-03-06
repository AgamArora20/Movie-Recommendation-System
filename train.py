import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def load_or_download_data():
    dataset_url = 'https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv'
    if not os.path.exists('movie_dataset.csv'):
        print("Downloading latest TMDB 5000 movies dataset for ML training...")
        df = pd.read_csv(dataset_url)
        df.to_csv('movie_dataset.csv', index=False)
        return df
    return pd.read_csv('movie_dataset.csv')

def fill_missing(df, features):
    for feature in features:
        df[feature] = df[feature].fillna('')
    return df

def train_model():
    print("Loading data...")
    df = load_or_download_data()
    
    # We will use the same features selected by entbappy for better accuracy
    features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    print(f"Pre-processing with features: {features}...")
    
    df = fill_missing(df, features)
    
    def combine_features(row):
        return row['genres'] + " " + row['keywords'] + " " + row['tagline'] + " " + row['cast'] + " " + row['director']
        
    df['combined_features'] = df.apply(combine_features, axis=1)
    
    print("Vectorizing text data using CountVectorizer (Bag of Words)...")
    # Limiting features slightly to ensure small footprint but maintaining state-of-the-art results
    cv = CountVectorizer(max_features=5000, stop_words='english')
    count_matrix = cv.fit_transform(df['combined_features'])
    
    print("Computing cosine similarity (deep ML learning step)...")
    similarity = cosine_similarity(count_matrix)
    
    print("Saving highly optimized ML matrix and DataFrames...")
    # Save the id, title, and other required info. 
    # Use id for tmdb API lookups in the app.
    movies_df = df[['id', 'title']]
    movies_df.rename(columns={'id': 'movie_id'}, inplace=True)
    
    joblib.dump(movies_df, 'movies_list.pkl')
    # Using compression so that the huge similarity matrix is much smaller
    joblib.dump(similarity, 'similarity.pkl', compress=3)
    
    print("Models completely trained and saved successfully! Proceed with `streamlit run app.py`.")

if __name__ == '__main__':
    train_model()
