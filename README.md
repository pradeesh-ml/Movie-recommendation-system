# Movie Recommendation System 🎬

A simple movie recommendation system built using **TF-IDF Vectorization** and **Cosine Similarity**. This project utilizes **Streamlit** for the web interface and provides movie recommendations based on content-based filtering.

## Features 🚀
- Recommend similar movies based on user input.
- Uses **Natural Language Processing (NLP)** techniques for movie matching.
- Web interface built with **Streamlit**.

## Installation 🔧
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender
   ```
   
## Dataset 📊
This project uses **tmdb_5000_movies.csv** and **tmdb_5000_credits.csv**, available in the `data/` folder.

## Usage ▶️
1. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
2. **Select a movie from the dropdown list** and get recommendations instantly!

## Project Structure 📁
```
movie_recommender/
│── data/                    # Dataset files
│── models/                  # Pretrained models
│── src/                     # Source code files
│   ├── model.py             # Recommendation logic
│   ├── recommend.py         # Streamlit app
│── README.md                # Project documentation
```

## Example Output 🖥️
After selecting a movie, the app provides 5 similar movie recommendations.

## Contributing 🤝
Feel free to submit issues or contribute to this project by creating pull requests!

## License 📜
This project is licensed under the MIT License.

