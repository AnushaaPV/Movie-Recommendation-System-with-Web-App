"""Build the CineMatch model files.

Run once:  python model_builder.py
Creates:   movies.pkl   (movie_id, title, tags)
           vectors.npz  (sparse word-count matrix, one row per movie)
"""
import ast
import os
import pickle
import sys

import pandas as pd
import scipy.sparse as sp
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

BASE = os.path.dirname(os.path.abspath(__file__))
MOVIES_CSV = os.path.join(BASE, "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(BASE, "tmdb_5000_credits.csv")
MOVIES_PKL = os.path.join(BASE, "movies.pkl")
VECTORS_NPZ = os.path.join(BASE, "vectors.npz")

ps = PorterStemmer()


def parse_names(text, limit=None, job=None):
    """Read a JSON-like string and return a list of 'name' values."""
    try:
        items = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    names = []
    for item in items:
        if job and item.get("job") != job:
            continue
        names.append(item["name"])
        if limit and len(names) == limit:
            break
    return names


def collapse(names):
    """'Sam Worthington' -> 'SamWorthington' so each person stays one token."""
    return [n.replace(" ", "") for n in names]


def stem(text):
    return " ".join(ps.stem(w) for w in text.split())


def main():
    for path in (MOVIES_CSV, CREDITS_CSV):
        if not os.path.exists(path):
            sys.exit(f"File not found: {path}")

    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    # Merge on the movie id (titles are not unique).
    credits = credits.rename(columns={"movie_id": "id"}).drop(columns=["title"])
    movies = movies.merge(credits, on="id")
    movies = movies[["id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies = movies.rename(columns={"id": "movie_id"}).dropna()
    movies = movies.drop_duplicates(subset="title").reset_index(drop=True)

    movies["overview"] = movies["overview"].apply(str.split)
    movies["genres"] = movies["genres"].apply(parse_names).apply(collapse)
    movies["keywords"] = movies["keywords"].apply(parse_names).apply(collapse)
    movies["cast"] = movies["cast"].apply(lambda t: collapse(parse_names(t, limit=3)))
    movies["crew"] = movies["crew"].apply(
        lambda t: collapse(parse_names(t, limit=1, job="Director"))
    )

    tags = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    movies["tags"] = tags.apply(lambda parts: stem(" ".join(parts).lower()))
    final = movies[["movie_id", "title", "tags"]]

    # Keep the matrix sparse: it is tiny compared to a dense 4800 x 4800 matrix.
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(final["tags"])

    with open(MOVIES_PKL, "wb") as f:
        pickle.dump(final, f)
    sp.save_npz(VECTORS_NPZ, vectors)

    # Quick check: Avatar must have TMDB id 19995.
    avatar = final.loc[final["title"] == "Avatar", "movie_id"].values
    print(f"Movies: {len(final)} | vectors: {vectors.shape} | Avatar id: {avatar}")
    if len(avatar) == 0 or int(avatar[0]) != 19995:
        sys.exit("Check failed: Avatar id should be 19995")
    print("Done. Now run: streamlit run app.py")


if __name__ == "__main__":
    main()
