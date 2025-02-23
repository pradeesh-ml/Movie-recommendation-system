import streamlit as st
import pickle

def load_data():
    with open(r"C:\Users\Pradeesh\Learning Zone\ML\movie recommender\models\cosinesimilarity.pkl", "rb") as f:
        similarity = pickle.load(f)
    with open(r"C:\Users\Pradeesh\Learning Zone\ML\movie recommender\models\dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    return dataset, similarity

def recommend(movie, dataset, similarity):
    index = dataset[dataset['title'] == movie].index[0]
    distances = similarity[index]
    movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [dataset.iloc[movie[0]].title for movie in movies]

dataset, similarity = load_data()
st.title("Movie Recommendation System")
st.header("Find Your Next Favorite Movie!")

movie_input = st.selectbox("Select a movie title:", dataset['title'].values)

if st.button("Get Recommendations"):
    recommendations = recommend(movie_input, dataset, similarity)
    for rec_movie in recommendations:
        st.write(rec_movie)


    
