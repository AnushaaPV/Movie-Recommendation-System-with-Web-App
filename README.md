# 🎬 CineMatch: Movie Recommendation System

A content-based movie recommender built with Python, scikit-learn and Streamlit. Pick a movie you liked and get similar ones, with posters and ratings from TMDB.

**Live demo:** https://movie-recommendation-system-with-web-app-5mjvhvknp97rbttmenihp.streamlit.app/
*(Free apps go to sleep. If you see a "wake up" button, click it and wait a few seconds.)*

## What it does
- Search 4,800+ movies from the TMDB 5000 dataset
- Recommends movies with similar plot, genres, keywords, cast and director
- Shows poster, year, rating and synopsis for each result
- You choose how many results to see (3 to 10)

## How it works
1. Join plot, genres, keywords, top 3 cast and the director into one text per movie
2. Join names into one word (`Sam Worthington` becomes `SamWorthington`) and reduce words to their roots with Porter stemming
3. Turn the text into word-count vectors (CountVectorizer, 5,000 words)
4. For the chosen movie, compute cosine similarity against all other movies and show the top N

The vectors are kept in sparse form and one similarity row is computed per request. This avoids storing a 4,806 x 4,806 matrix.

## Tech stack
Python, pandas, scikit-learn, NLTK, SciPy, Streamlit, TMDB API

## Run it yourself
```bash
git clone https://github.com/AnushaaPV/Movie-Recommendation-System-with-Web-App.git
cd Movie-Recommendation-System-with-Web-App
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python model_builder.py         # builds movies.pkl and vectors.npz (about 1 minute)
streamlit run app.py
```

### TMDB API key (optional, for posters and ratings)
Get a free key at themoviedb.org (Settings, then API). Then **do not put it in the code**. Create `.streamlit/secrets.toml`:
```toml
TMDB_API_KEY = "your_key_here"
```
On Streamlit Cloud, add the same line under App settings, then Secrets. Without a key the app still works, but posters are hidden.

## Project files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app: search, recommend, show results |
| `model_builder.py` | Builds the model files from the two CSVs |
| `movies.pkl`, `vectors.npz` | Generated model files (small, committed so the app starts fast) |
| `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv` | Dataset |

## Limits
- Only uses movie content. It does not use user ratings or watch history.
- Results depend on the text in the dataset, so very new movies are not included.

## What I learned
- Turning nested JSON-like text into model features
- Content-based vs collaborative filtering
- Why sparse matrices save memory
- Hiding API keys with secrets and caching API calls
- Deploying a Streamlit app

## Credits
- Dataset: [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- This product uses the TMDB API but is not endorsed or certified by TMDB.
