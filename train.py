import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

def train_model(data_path='movies.csv'):
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found. Please provide it.")
        print("Creating a sample movie dataset for demonstration...")
        data = {
            'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Toy Story', 'Batman Begins', 'The Matrix Reloaded'],
            'overview': [
                'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
                'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
                'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
                'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
                'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy\'s room.',
                'After training with his mentor, Batman begins his fight to free crime-ridden Gotham City from corruption.',
                'Neo and the rebel leaders estimate that they have 72 hours until 250,000 probes discover Zion and destroy it and its inhabitants.'
            ],
            'movie_id': [1, 2, 3, 4, 5, 6, 7]
        }
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    if 'tags' in df.columns:
        text_feature = 'tags'
    elif 'overview' in df.columns:
        text_feature = 'overview'
    else:
        text_cols = df.select_dtypes(include=['object']).columns
        df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
        text_feature = 'combined_text'

    df[text_feature] = df[text_feature].fillna('')
    
    print("Vectorizing text data...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df[text_feature])
    
    print("Computing cosine similarity...")
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    print("Saving model artifacts...")
    df_to_save = df[['title']] if 'movie_id' not in df.columns else df[['movie_id', 'title']]
    joblib.dump(df_to_save, 'movies_list.pkl')
    joblib.dump(similarity, 'similarity.pkl')
    
    print("Training complete! Model artifacts saved as movies_list.pkl and similarity.pkl")

if __name__ == '__main__':
    train_model()
