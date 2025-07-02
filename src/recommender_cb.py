import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.product_ids = []
        self.tfidf_matrix = None

    def fit(self, df: pd.DataFrame, text_col: str, id_col: str):
        self.product_ids = df[id_col].tolist()
        tfidf_matrix = self.tfidf.fit_transform(df[text_col])
        self.tfidf_matrix = tfidf_matrix
        self.df = df
        self.id_col = id_col
        return self

    def recommend(self, product_id: str, n=5):
        if product_id not in self.product_ids:
            return pd.DataFrame()

        idx = self.product_ids.index(product_id)
        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(cosine_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        top_indices = [i for i, _ in sim_scores[1:n+1]]
        results = self.df.iloc[top_indices][[self.id_col]].copy()
        results['similarity_score'] = [cosine_sim[i] for i in top_indices]
        return results
