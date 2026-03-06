# Movie Recommendation System 🎬

A Machine Learning based recommendation engine that suggests movies based on user preferences. This project implements similarity-based filtering using content metadata and provides an interactive visualization web app built with Streamlit.

## Resume Highlights
- **Developed a machine learning-based recommendation engine** suggesting movies based on user preferences.
- **Implemented similarity-based filtering** utilizing TF-IDF Vectorization and Cosine Similarity, and developed an **interactive visualization through Streamlit**.

## Features
- **Content-Based Filtering:** Recommends movies similar to the one selected by the user by analyzing text features (overview, genres, or tags) to find similar content.
- **Interactive UI:** A highly interactive web application built with Streamlit, where users can browse and find movie recommendations seamlessly.
- **Scalable Architecture:** Pre-computes and stores TF-IDF features and cosine similarity matrices using `joblib` for rapid retrieval.

## Tech Stack
- **Python**
- **Pandas** for data manipulation and analysis
- **Scikit-learn** for natural language processing and machine learning algorithms
- **Streamlit** for front-end interface and interactive visualization

## How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd Movie-Recommendation-System
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**
   Run the following script to generate content-based embeddings and similarity matrices (`movies_list.pkl`, `similarity.pkl`).
   If a custom `movies.csv` dataset is not found, a sample dataset will be generated automatically.
   ```bash
   python train.py
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Future Scope / Extensibility
While this project focuses on ML with Python and Streamlit for the main functionality, it can easily be extended into a full MERN/PERN stack web application with React.js for more complex UI features and MongoDB/Node.js to handle user authentication, storing favorite movies, and managing user profiles.
