"""CineMatch - content-based movie recommender (Streamlit)."""
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import requests
import scipy.sparse as sp
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch - Movie Recommender", page_icon="🎬", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
MOVIES_PKL = os.path.join(BASE, "movies.pkl")
VECTORS_NPZ = os.path.join(BASE, "vectors.npz")

# The key comes from Streamlit secrets or an environment variable. Never from the code.
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

NO_INFO = {"poster_url": None, "rating": "N/A", "overview": "No description available.", "year": "N/A"}


@st.cache_resource
def load_model():
    if not (os.path.exists(MOVIES_PKL) and os.path.exists(VECTORS_NPZ)):
        st.error("Model files not found. Run `python model_builder.py` first.")
        st.stop()
    with open(MOVIES_PKL, "rb") as f:  # our own file, created by model_builder.py
        movies_df = pickle.load(f)
    vectors = sp.load_npz(VECTORS_NPZ).tocsr()
    return movies_df, vectors


movies_df, vectors = load_model()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_info(movie_id: int) -> dict:
    """Get poster, rating, year and overview from TMDB. Cached for one day."""
    if not TMDB_API_KEY:
        return NO_INFO
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10,
        )
        r.raise_for_status()
    except requests.RequestException:
        return NO_INFO
    data = r.json()
    poster = data.get("poster_path")
    rating = data.get("vote_average")
    date = data.get("release_date") or ""
    return {
        "poster_url": f"https://image.tmdb.org/t/p/w500{poster}" if poster else None,
        "rating": round(rating, 1) if rating else "N/A",
        "overview": data.get("overview") or NO_INFO["overview"],
        "year": date[:4] or "N/A",
    }


def recommend(row_index: int, n: int = 6):
    """Return the row positions of the n most similar movies."""
    scores = cosine_similarity(vectors[row_index], vectors).ravel()
    ranked = scores.argsort()[::-1]
    return [int(i) for i in ranked if i != row_index][:n]


# ---------------- UI ----------------
st.markdown(
    "<h1 style='text-align:center; color:#E50914;'>🎬 CineMatch</h1>"
    f"<p style='text-align:center; color:gray;'>Content-based movie recommender · {len(movies_df):,} movies</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

if not TMDB_API_KEY:
    st.info("TMDB key not set, so posters and ratings are hidden. Recommendations still work.")

titles = movies_df["title"].tolist()
selected = st.selectbox("🔍 Search for a movie you liked:", options=titles, index=None,
                        placeholder="Start typing a movie name...")
num_recs = st.slider("Number of recommendations", 3, 10, 6)

if st.button("🎯 Find Similar Movies", type="primary", use_container_width=True):
    if not selected:
        st.warning("Please select a movie first.")
    else:
        row = titles.index(selected)
        rec_rows = recommend(row, num_recs)
        rec_ids = [int(movies_df.iloc[i]["movie_id"]) for i in rec_rows]

        with ThreadPoolExecutor(max_workers=5) as pool:  # fetch posters in parallel
            infos = list(pool.map(fetch_info, rec_ids))

        st.markdown(f"### 🎬 Because you liked **{selected}**:")
        for start in range(0, len(rec_rows), 3):
            cols = st.columns(3)
            for col, i, info in zip(cols, rec_rows[start:start + 3], infos[start:start + 3]):
                with col:
                    if info["poster_url"]:
                        st.image(info["poster_url"], use_container_width=True)
                    else:
                        st.caption("No poster available")
                    st.markdown(f"**{movies_df.iloc[i]['title']}**")
                    st.markdown(f"📅 {info['year']}  ⭐ {info['rating']}/10")
                    with st.expander("📖 Synopsis"):
                        text = info["overview"]
                        st.write(text[:300] + "..." if len(text) > 300 else text)

with st.expander("🔧 How does CineMatch work?"):
    st.markdown(
        """
| Step | What happens |
|---|---|
| Features | Plot, genres, keywords, top 3 cast and director are joined into one text |
| Cleaning | Names are joined (SamWorthington) and words are reduced to roots |
| Vectors | CountVectorizer turns the text into 5,000-word count vectors |
| Similarity | Cosine similarity between the chosen movie and all others |
| Ranking | The top N most similar movies are shown |
"""
    )

st.markdown("---")
st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB. "
           "Data: TMDB 5000 Movie Dataset (Kaggle).")
