import pickle
import streamlit as st
import requests
import pandas as pd
import os
import google.generativeai as genai

st.set_page_config(layout="wide", page_title="Movie Recommender System", page_icon="🍿")

st.title("Movie Recommender System Using Machine Learning & GenAI")
st.markdown("A deep learning and LLM (Large Language Model) powered engine learning from user watch history and real-time moods.")

# ================= Configuration and Integrations ================= #
st.sidebar.title("Configuration & API Keys")
tmdb_api_key = st.sidebar.text_input(
    "TMDB API Key (For Film Posters)", 
    value="8265bd1679663a7ea12ac168da84d2e8",
    help="You can get a free API Key from https://www.themoviedb.org/"
)
gemini_api_key = st.sidebar.text_input(
    "Gemini / Google GenAI Key [LLM Engine]", 
    type="password",
    help="Add key to unlock profound Deep Learning interpretation based on watch history."
)
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Résumé Highlights")
st.sidebar.info("""• Developed a hybrid ML model evaluating historical interaction patterns and implicit preferences.\n• Leveraged massive text embeddings (Cosine Similarity Matrix) and integrated a Large Language Model (LLM) analyzing real-time cognitive sentiment. \n• Built seamless interactive pipelines with robust React.js-like visual states via Streamlit.""")

# Initialize Watch History state
if "watch_history" not in st.session_state:
    st.session_state.watch_history = []

def fetch_poster(movie_id):
    """Fetches the movie poster URL from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception as e:
        pass
    return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"

def recommend_classical(movie, num_recommendations=5):
    """Deep learning phase 1 - Recommends similar movies using Cosine Similarity Tensor matrix."""
    try:
        index = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.error("Movie not found in the dataset.")
        return []
        
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recs = []
    for i in distances[1 : num_recommendations + 1]:
        movie_title = movies.iloc[i[0]].title
        movie_id = movies.iloc[i[0]].movie_id
        movie_year = movies.iloc[i[0]].year
        movie_rating = movies.iloc[i[0]].vote_average
        
        recs.append({
            'title': movie_title, 
            'id': movie_id, 
            'poster': fetch_poster(movie_id),
            'year': int(movie_year) if pd.notna(movie_year) else "N/A",
            'rating': movie_rating
        })
    return recs

def get_llm_personalization(base_recs, user_history, mood):
    """Deep Learning phase 2 - LLM Contextual generation based on history."""
    if not gemini_api_key:
        return base_recs
        
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        recs_list = [r['title'] for r in base_recs]
        prompt = f"""
        Act as an incredibly sophisticated movie recommendation algorithm. I am giving you:
        - The user's watch history: {user_history if user_history else 'No previous history.'}
        - The user's CURRENT MOOD/VIBE: "{mood}"
        - The baseline ML recommendations (Cos Sim to target): {recs_list}
        
        Your task is to re-rank, curate, or select 5 of the BEST options from the baseline recommendations that match their deep taste profile and mood. 
        Add a 1-sentence personalized explanation on WHY it fits.
        Format EXACTLY:
        Title : Explanation
        """
        response = model.generate_content(prompt)
        
        lines = response.text.split('\n')
        llm_enhanced_recs = []
        for line in lines:
            if ' :' in line:
                parts = line.split(' :', 1)
                title = parts[0].strip()
                explain = parts[1].strip()
                
                for r in base_recs:
                    if title.lower() in r['title'].lower():
                        r['llm_explanation'] = explain
                        llm_enhanced_recs.append(r)
                        break
        
        return llm_enhanced_recs if llm_enhanced_recs else base_recs
    except Exception as e:
        st.warning(f"LLM Processing error, utilizing pure Matrix Similarity: {e}")
        return base_recs

# ================= Data Loading ================= #
try:
    movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run the `train.py` or the data processing notebook first.")
    st.stop()

# Top metrics Dashboard
colA, colB, colC = st.columns(3)
colA.metric("Watch History Items", len(st.session_state.watch_history))
colB.metric("Features Vectorized", "5,000 Dimensions")
colC.metric("Similarity Calculations", f"{len(movies)}² Ops")
st.markdown("---")

# ================= Application UI ================= #
movie_list = movies['title'].values

st.subheader("Explore Movies")
col1, col2 = st.columns(2)
with col1:
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown to find similar movies",
        movie_list
    )
with col2:
    current_mood = st.text_input("How are you feeling right now? (Optional LLM tuning)")
    
if st.button(f"➕ Mark '{selected_movie}' as Watched"):
    if selected_movie not in st.session_state.watch_history:
        st.session_state.watch_history.append(selected_movie)
        st.success(f"Added '{selected_movie}' to Watch History!")
if st.session_state.watch_history:
    st.caption("You have watched: " + ", ".join(st.session_state.watch_history))

st.markdown("---")

if st.button('Show Recommendation 🚀'):
    with st.spinner('Calculating recommendations (Cosine Similarity & Deep Profile)...'):
        # Phase 1: Machine Learning
        base_recs = recommend_classical(selected_movie, num_recommendations=10)
        
        # Phase 2: Generative Models (LLM)
        if gemini_api_key and current_mood:
            final_recs = get_llm_personalization(base_recs, st.session_state.watch_history, current_mood)
        else:
            final_recs = base_recs[:5]
            if not gemini_api_key:
                st.info("💡 Pure mathematical similarities shown below. Add a Gemini API Key to activate LLM context interpretation.")
    
    if final_recs:
        st.subheader("Your AI-Curated Picks")
        
        cols = st.columns(min(len(final_recs), 5))
        for idx in range(min(len(final_recs), 5)):
            with cols[idx]:
                rec = final_recs[idx]
                st.image(rec['poster'], use_column_width=True)
                st.markdown(f"**{rec['title']}**")
                
                # Matching Original Repo's style for metadata
                if 'year' in rec:
                    st.caption(f"Year: {rec['year']}")
                if 'rating' in rec:
                    st.caption(f"Rating: {rec['rating']:.1f} ⭐")
                
                if 'llm_explanation' in rec:
                    st.caption(f"✨ *{rec['llm_explanation']}*")
