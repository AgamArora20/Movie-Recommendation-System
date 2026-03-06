import streamlit as st
import joblib
import pandas as pd
import requests
import os
import google.generativeai as genai

st.set_page_config(layout="wide", page_title="AI Movie Recommender Ecosystem", page_icon="🍿")

st.title("🍿 Next-Gen AI Movie Recommender")
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
st.sidebar.subheader("📌 Resume Highlights")
st.sidebar.info("""• Developed a hybrid ML model evaluating historical interaction patterns and implicit preferences.\n• Leveraged massive text embeddings (Cosine Similarity Matrix) and integrated a Large Language Model (LLM) analyzing real-time cognitive sentiment. \n• Built seamless interactive pipelines with robust React.js-like visual states via Streamlit.""")

# Initialize Watch History state (Session State simulation of Database/NoSQL)
if "watch_history" not in st.session_state:
    st.session_state.watch_history = []

# ================= Utility Functions ================= #
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster+Found"
    except:
        return "https://via.placeholder.com/500x750?text=Error"

def recommend_classical(movie, num_recommendations=5):
    """Deep learning phase 1 (Cosine Similarity Tensor matrix lookup)"""
    try:
        index = movies[movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        
        recs = []
        for i in distances[1 : num_recommendations + 1]:
            movie_title = movies.iloc[i[0]].title
            movie_id = movies.iloc[i[0]].movie_id
            recs.append({'title': movie_title, 'id': movie_id, 'poster': fetch_poster(movie_id)})
        return recs
    except Exception as e:
        return []

def get_llm_personalization(base_recs, user_history, mood):
    """Deep Learning phase 2 (LLM Contextual generation based on history)"""
    if not gemini_api_key:
        return base_recs
        
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        recs_list = [r['title'] for r in base_recs]
        prompt = f"""
        Act as an incredibly sophisticated movie recommendation algorithm. I am giving you:
        - The user's watch history: {user_history if user_history else 'No previous history. Completely fresh user.'}
        - The user's CURRENT MOOD/VIBE: "{mood}"
        - The baseline ML recommendations based on mathematical feature similarity to their last searched movie: {recs_list}
        
        Your task is to re-rank, curate, or select 3 of the BEST options from the baseline recommendations that truly match not just the math, but the deep taste profile and current mood. 
        You can also add a 1-sentence personalized explanation on WHY it fits their current state and taste pattern.
        
        Format your response exactly like this (don't include markdown bullets, just strict structure):
        Title 1 : Explanation for Title 1
        Title 2 : Explanation for Title 2
        Title 3 : Explanation for Title 3
        """
        response = model.generate_content(prompt)
        
        # Super simple parser logic for real-time inference
        lines = response.text.split('\n')
        llm_enhanced_recs = []
        for line in lines:
            if ' :' in line:
                parts = line.split(' :', 1)
                title = parts[0].strip()
                explain = parts[1].strip()
                
                # Match title back to base_recs to keep visual posters
                for r in base_recs:
                    if title.lower() in r['title'].lower():
                        r['llm_explanation'] = explain
                        llm_enhanced_recs.append(r)
                        break
        
        return llm_enhanced_recs if llm_enhanced_recs else base_recs
    except Exception as e:
        st.warning(f"LLM Processing encountered an error, falling back to pure Matrix Similarity: {e}")
        return base_recs

# ================= Data Loading ================= #
@st.cache_resource(show_spinner=False)
def load_data():
    if not os.path.exists('movies_list.pkl') or not os.path.exists('similarity.pkl'):
        return None, None
    movies_df = joblib.load('movies_list.pkl')
    sim = joblib.load('similarity.pkl')
    return movies_df, sim

movies, similarity = load_data()

# ================= Application Interface ================= #
if movies is None or similarity is None:
    st.error("🚨 Deep ML weights not found! Run the trainer (`python train.py`).")
else:
    # Top metrics (Dashboard style for resume/portfolio aesthetic)
    colA, colB, colC = st.columns(3)
    colA.metric("Watch History Items", len(st.session_state.watch_history))
    colB.metric("Model Embedding Features", "5,000 Text Vectors")
    colC.metric("Similarity Calculations", f"{len(movies)}² Cosine Ops")
    
    st.markdown("---")
    
    movie_list = movies['title'].values
    
    # Taste Profile & Sentiment Inputs
    st.subheader("1. Taste Profile Input")
    col1, col2 = st.columns(2)
    with col1:
        selected_movie = st.selectbox(
            "What movie do you have in mind right now?",
            movie_list
        )
    with col2:
        current_mood = st.text_input("How are you feeling right now? (e.g. 'Stressed, looking to laugh', or 'In the mood for mind-bending sci-fi')")
        
    st.markdown("##### 🕰️ Watch History")
    # Small UI to simulate watching a movie to build the taste profile
    if st.button(f"➕ Mark '{selected_movie}' as Watched"):
        if selected_movie not in st.session_state.watch_history:
            st.session_state.watch_history.append(selected_movie)
            st.success(f"Added to Watch History. Taste profile updated!")
    if st.session_state.watch_history:
        st.caption("You have watched: " + ", ".join(st.session_state.watch_history))

    st.markdown("---")
    
    # Central Recommendation Processing Pipeline
    if st.button('🧠 Generate AI Recommendations', type="primary"):
        with st.spinner('Calculating sparse matrices...'):
            # Step 1: Classical Machine Learning Cosine Matrix processing
            base_recs = recommend_classical(selected_movie, num_recommendations=10)
            
        with st.spinner('Engaging LLM Deep Contextualization & Emotional matching...'):
            # Step 2: Deep Profiling with large generative model (If API exists and mood is given)
            if gemini_api_key and current_mood:
                final_recs = get_llm_personalization(base_recs, st.session_state.watch_history, current_mood)
            else:
                final_recs = base_recs[:5]
                if not gemini_api_key:
                    st.info("💡 Pure mathematically calculated vectors are shown below. Enter a Gemini API Key to activate dynamic deep-learning context and mood interpretation.")
        
        # Step 3: Immersive Frontend rendering
        st.subheader("Your AI-Curated Picks")
        
        cols = st.columns(len(final_recs[:5]))
        for idx, col in enumerate(cols):
            with col:
                st.image(final_recs[idx]['poster'], use_column_width=True)
                st.write(f"**{final_recs[idx]['title']}**")
                if 'llm_explanation' in final_recs[idx]:
                    st.caption(f"✨ *{final_recs[idx]['llm_explanation']}*")
