# Movie Recommender System Using Machine Learning & GenAI!

Recommendation systems are becoming increasingly important in today's extremely busy world. People are always short on time with the myriad tasks they need to accomplish in the limited 24 hours. Therefore, the recommendation systems are important as they help them make the right choices, without having to expend their cognitive resources.

The purpose of a recommendation system basically is to search for content that would be interesting to an individual. Moreover, it involves a number of factors to create personalised lists of useful and interesting content specific to each user/individual. Recommendation systems are Artificial Intelligence based algorithms that skim through all possible options and create a customized list of items that are interesting and relevant to an individual. These results are based on their profile, search/browsing history, what other people with similar traits/demographics are watching, and how likely are you to watch those movies. This is achieved through predictive modeling and heuristics with the data available.

## Types of Recommendation System :

### 1 ) Content Based :
Content-based systems, which use characteristic information and takes item attributes into consideration.
These systems make recommendations using a user's item and profile features. They hypothesize that if a user was interested in an item in the past, they will once again be interested in it in the future.
One issue that arises is making obvious recommendations because of excessive specialization.

### 2 ) Collaborative Based :
Collaborative filtering systems, which are based on user-item interactions.
Clusters of users with same ratings, similar users. Book recommendation, so use cluster mechanism. We take only one parameter, ratings or comments.
In short, collaborative filtering systems are based on the assumption that if a user likes item A and another user likes the same item A as well as another item, item B, the first user could also be interested in the second item.

### 3 ) Hybrid Based :
Hybrid systems, which combine both types of information with the aim of avoiding problems that are generated when working with just one kind.
Combination of both and used nowadays. Uses: word2vec, embedding.

---

## About this project:
This is a streamlit web application that can recommend various kinds of similar movies based on an user interest. I have expanded upon the base Content-Based filtering by integrating a **Hybrid approach using Large Language Models (LLMs)**. The system analyzes user Watch History and real-time Mood using Google Gemini APIs atop the Classical ML Tensor space (Cosine Similarity Matrix).

### Dataset used:
Yes, this project uses the exact high-quality dataset:
[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

### Resume Highlights / What was added:
- **Developed a hybrid ML model** evaluating historical interaction patterns and implicit preferences.
- **Leveraged massive text embeddings** (Cosine Similarity Matrix) and integrated a Large Language Model (LLM) analyzing real-time cognitive sentiment.
- **Built seamless interactive pipelines** with robust React.js-like visual states via Streamlit.

### Concept used to build the model:
1. **Cosine Similarity** is a metric that allows you to measure the similarity of the documents.
2. In order to demonstrate cosine similarity function we need vectors. Here vectors are numpy arrays derived from text features (bag-of-words using CountVectorizer).
3. Finally, Once we have vectors, We can call `cosine_similarity()` by passing both vectors. It will calculate the cosine similarity between these two.
4. It will be a value between [0,1]. If it is 0 then both vectors are completely different. But if it is 1, they are completely similar.
5. For more details, check URL: https://www.learndatasci.com/glossary/cosine-similarity/

---

## How to run?

**STEPS:**
Clone the repository:
```bash
git clone https://github.com/AgamArora20/Movie-Recommendation-System.git
```

**STEP 01 - Create environment after opening the repository**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**STEP 02 - Install the requirements**
```bash
pip install -r requirements.txt
```

**STEP 03 - Run this file to generate the initial ML base models**
Make sure the dataset is loaded or it will auto-download the required data to build `artifacts/`.
```bash
python train.py
```

**(Alternative: Run the Data Analysis Jupyter Notebook if preferred)**
```bash
jupyter notebook "Movie Recommender System Data Analysis.ipynb"
```

**STEP 04 - Run the Streamlit Application**
```bash
streamlit run app.py
```

### Author:
**Agam Arora** 
Data Scientist & ML Engineer  
GitHub: [AgamArora20](https://github.com/AgamArora20)
