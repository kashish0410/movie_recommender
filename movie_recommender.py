import os
import requests
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
hf_token = os.getenv("gemini_api_key")

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
headers = {"X-goog-api-key": f"{hf_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/kkash/OneDrive/Desktop/py/movie_recommender/IMDb_All_Genres_etf_clean1.csv")
    df['genre_list'] = df['side_genre'].apply(lambda s: [g.strip() for g in s.split(",")] if isinstance(s, str) else [])
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(mlb.fit_transform(df['genre_list']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df, genre_encoded], axis=1)
    df['cluster'] = KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(genre_encoded)
    return df, mlb

df, mlb = load_data()

# ---------------------------
# Recommendation Functions
# ---------------------------
def recommend_similar_movies(movie_title, n=5):
    if movie_title not in df['Movie_Title'].values:
        return []
    cl = df.loc[df['Movie_Title'] == movie_title, 'cluster'].iloc[0]
    recs = df[(df['cluster'] == cl) & (df['Movie_Title'] != movie_title)]
    return recs[['Movie_Title', 'genre_list']].head(n)

def recommend_by_prompt(prompt, n=5):
    prompt = prompt.lower()
    matched = [g for g in mlb.classes_ if g.lower() in prompt]
    if not matched:
        return f"No matching genres for '{prompt}'."
    filtered = df.copy()
    for g in matched:
        filtered = filtered[filtered[g] == 1]
    return filtered[['Movie_Title', 'genre_list']].head(n) if not filtered.empty else "No matches found."

def ask_gemini(user_query, n=5):
    sample = ", ".join(df['Movie_Title'].sample(30, random_state=42))
    prompt = f"You are a movie recommendation assistant.\nMovies: {sample}.\nUser wants: {user_query}.\nRecommend {n} movies from the list, comma-separated."

    result = query({
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    })

    try:
        text = result['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        return []

    dataset_titles = set(df['Movie_Title'])
    return [m.strip() for m in text.split(",") if m.strip() in dataset_titles]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 Movie Recommender System")

# By Movie Title
st.subheader("🔎 Recommend by Movie Title")
movie = st.selectbox("Choose a movie:", sorted(df['Movie_Title']))
if st.button("Get Recommendations (By Movie)"):
    recs = recommend_similar_movies(movie)
    if getattr(recs, 'empty', False):
        st.warning("No similar movies found.")
    else:
        st.table(recs)

# By Genre
st.subheader("🎭 Recommend by Genre")
genre_prompt = st.text_input("Enter genres (e.g., Action, Drama):")
if st.button("Find by Genre"):
    res = recommend_by_prompt(genre_prompt)
    st.warning(res) if isinstance(res, str) else st.table(res)

# By Gemini
st.subheader("🤖 AI-Powered (Gemini) Recommendation")
user_query = st.text_area("What kind of movie do you want?")
if st.button("Ask Gemini"):
    if user_query.strip():
        with st.spinner("Gemini is thinking..."):
            recs = ask_gemini(user_query)
        if recs:
            for i, m in enumerate(recs, 1):
                st.write(f"{i}. {m}")
        else:
            st.warning("Gemini gave no recommendations.")
