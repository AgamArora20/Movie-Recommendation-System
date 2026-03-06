# AI Movie Recommender Ecosystem 🍿

An innovative movie recommendation engine that bridges the gap between traditional Machine Learning and Large Language Models (LLM). 

People today are overwhelmed by choices across multiple streaming platforms. This project solves that "cognitive overload" by analyzing thousands of cinematic features (genres, cast, blurbs) alongside your **dynamic mood** and **personal watch history** to curate the perfect movie night.

---

## 🚀 The Innovation: A Hybrid Architecture

Most recommendation systems rely solely on what you've watched in the past. This project introduces a **two-phase hybrid filtering approach**:

### 1. The Deep Learning Foundation (Content-Based)
Using the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), the engine maps every movie into a high-dimensional mathematical space.
- It parses attributes like `genres`, `keywords`, `tagline`, `cast`, and `director`.
- These features are vectorized (Bag of Words / CountVectorizer).
- A **Cosine Similarity Matrix** is computed to calculate the exact spatial distance between every single movie, surfacing statistically identical content.

### 2. The GenAI Mood & Context Engine (LLM-Based)
Math isn't everything. Sometimes a movie mathematically matches, but it doesn't fit your *vibe*.
- **State Management:** The app tracks your real-time **Watch History**.
- **Generative Evaluation:** By integrating **Google Gemini / LLM APIs**, the system takes your mathematical recommendations, reads your current textual mood (e.g., *"I'm stressed and need something light"*), evaluates your watch history, and **re-ranks** the films.
- It outputs exactly *why* a movie fits your current state to eliminate choice paralysis.

---

## 🛠️ Features & Resume Highlights

- **Developed a hybrid ML model** evaluating historical interaction patterns and implicit preferences.
- **Leveraged massive text embeddings** (Cosine Similarity Matrix) and integrated a Large Language Model (LLM) analyzing real-time cognitive sentiment.
- **Built an immersive UI pipeline** with robust state management and live API connections (TMDB Posters) via Streamlit.

---

## 💻 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/AgamArora20/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

**2. Set up your environment**
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
# source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Generate ML Artifacts**
Before running the UI, you must bake the mathematical models. Running this script will dynamically build the `similarity.pkl` and `movie_dict.pkl` matrices.
```bash
python train.py
```

**5. Launch the Ecosystem**
```bash
streamlit run app.py
```
*(Once launched, insert your TMDB and Google API keys in the sidebar to unlock live posters and GenAI mood analysis!)*

---

### Author
**Agam Arora**  
[GitHub Profile](https://github.com/AgamArora20)
