import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

def load_or_download_data():
    dataset_url = 'https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv'
    if not os.path.exists('movie_dataset.csv'):
        print("Downloading dataset...")
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
    
    # Also want release_date to extract year, and vote_average
    df['release_date'] = df['release_date'].fillna('1900-01-01')
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['year'] = df['year'].fillna(0).astype(int)
    
    df['vote_average'] = df['vote_average'].fillna(0.0)
    
    df = fill_missing(df, features)
    
    def combine_features(row):
        return row['genres'] + " " + row['keywords'] + " " + row['tagline'] + " " + row['cast'] + " " + row['director']
        
    df['combined_features'] = df.apply(combine_features, axis=1)
    
    print("Vectorizing text data using CountVectorizer...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    count_matrix = cv.fit_transform(df['combined_features'])
    
    print("Computing cosine similarity...")
    similarity = cosine_similarity(count_matrix)
    
    print("Saving ML metrics to artifacts...")
    movies_df = df[['id', 'title', 'year', 'vote_average']]
    movies_df.rename(columns={'id': 'movie_id'}, inplace=True)
    
    os.makedirs('artifacts', exist_ok=True)
    
    # To match his structure, we save as a dictionary using pickle
    pickle.dump(movies_df.to_dict(), open('artifacts/movie_dict.pkl', 'wb'))
    pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))
    
    print("Models computed and saved in artifacts/")

if __name__ == '__main__':
    train_model()
