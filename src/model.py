import pandas as pd
import pickle
import ast
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
def load_data(movies_path, credits_path):
    movies_data = pd.read_csv(movies_path)
    credits_data= pd.read_csv(credits_path)
    credits_data.rename(columns={'movie_id': 'id'}, inplace=True)
    df = movies_data.merge(credits_data, on='id')
    df = df[['genres', 'keywords', 'original_title', 'cast', 'crew', 'overview']]
    df.dropna(inplace=True)
    df.rename(columns={'original_title': 'title'}, inplace=True)
    return df

# Feature extraction functions
def extract_genre(col):
    return [i['name'] for i in ast.literal_eval(col)]

def extract_director(col):
    return [i['name'] for i in ast.literal_eval(col) if i['job'] == "Director"]

def extract_cast(col):
    return [i['name'] for i in ast.literal_eval(col)[:5]]

def preprocess_data(df):
    df['genres'] = df['genres'].apply(extract_genre)
    df['keywords'] = df['keywords'].apply(extract_genre)
    df['crew'] = df['crew'].apply(extract_director)
    df['cast'] = df['cast'].apply(extract_cast)
    df['overview'] = df['overview'].apply(lambda x: x.split())
    df['corpus'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
    df = df[['title', 'corpus']]
    df['corpus'] = df['corpus'].apply(lambda x: " ".join(x).lower())
    return df

# Apply stemming
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

def vectorize_and_save(df, model_path, similarity_path):
    df['corpus'] = df['corpus'].apply(stem)
    tfidf = TfidfVectorizer(stop_words='english')
    tfid_matrix = tfidf.fit_transform(df['corpus'])
    similarity = cosine_similarity(tfid_matrix)
    
    with open(model_path, "wb") as f:
        pickle.dump(df, f)
    with open(similarity_path, "wb") as f:
        pickle.dump(similarity, f)
    
    return df, similarity

def recommend(movie, df, similarity):
    if movie not in df['title'].values:
        return "Movie not found."
    
    index = df[df['title'] == movie].index[0]
    distances = similarity[index]
    movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:5]
    
    return [df.iloc[i[0]].title for i in movies]

if __name__ == "__main__":
    df = load_data(r"C:\Users\Pradeesh\Learning Zone\ML\movie recommender\data\tmdb_5000_movies.csv", r"C:\Users\Pradeesh\Learning Zone\ML\movie recommender\data\tmdb_5000_credits.csv")
    df = preprocess_data(df)
    df, similarity = vectorize_and_save(df, r"C:\Users\Pradeesh\Learning Zone\ML\movie recommender\models\dataset.pkl", r"C:\Users\Pradeesh\Learning Zone\ML\movie recommender\models\cosinesimilarity.pkl")
    
    movie_name = "The Matrix"
    recommendations = recommend(movie_name, df, similarity)
    print("Recommendations for:", movie_name)
    print(recommendations)
