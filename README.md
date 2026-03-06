# 🎬 Next-Gen AI Movie Recommender �

A Machine Learning and LLM-powered movie recommendation engine. It not only uses traditional Deep Learning algorithms (Cosine Similarity on vectorized metadata) but also integrates Large Language Models (LLM) to adapt dynamically to a user's **Watch History**, **Performance (implicit interactions)**, and their **Current Mood**.

## Resume Highlights
- **Developed a hybrid machine learning recommendation engine** suggesting movies based on deep user preferences, watch history, and behavioral performance.
- **Integrated Large Language Models (LLM)** to process extreme conversational context, dynamically matching content to the user's real-time mood and past consumption taste profile.
- **Implemented similarity-based filtering** utilizing advanced Natural Language Processing feature vectorization (CountVectorizer / TF-IDF) and Cosine Similarity, deploying the interactive model pipeline via a **Streamlit and React.js-inspired UI architecture**.
- **Leveraged robust Deep Learning concepts and API orchestrations** to fetch real-time metadata (posters, genres) through integration with platforms like TMDB and generative AI logic for context-aware synthetic reasoning.

## How the Hybrid Architecture Works:
1. **Classical ML Sub-Engine (Deep Feature Extraction)** 🧠
   - Uses vectorized bag-of-words / TF-IDF representations of `genres`, `keywords`, `cast`, `director`, and `taglines` to map all media content into a vast high-dimensional tensor space.
   - Calculates the **Cosine Similarity distance** (deep proximity) of every movie against every other movie to find strictly similar stylistic/thematic peers.
2. **LLM & Behavioral Profiling Engine** 🤖
   - Tracks **Watch History** and general taste profiles using session management.
   - Consumes the **Current User Mood** (e.g., "I'm feeling very sad tonight", or "Hyped for some action!") and contextually aligns it with the classical recommendations using advanced synthetic reasoning via Google Gemini API / LLMs.

## How to Run Locally

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd Movie-Recommendation-System
   ```

2. **Create a virtual environment (optional but highly recommended)**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML Base Model**
   Run the training architecture script to dynamically download the vast original dataset and bake the massive sparse matrices.
   ```bash
   python train.py
   ```

5. **Run the Front-End Application**
   ```bash
   streamlit run app.py
   ```

## Setup API Keys within the App
Once you launch the Streamlit App (`app.py`), you might want to insert:
- **TMDB API Key**: For fetching high-res cinematic movie posters.
- **Google Gemini / LLM Token (Optional)**: To unlock the "Mood AI" analysis and pure generative contextualization based on watch history.
