import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, product_df):
        self.products = product_df.copy()
        self.products['product_category_name'] = self.products['product_category_name'].fillna('unknown')
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.products['product_category_name'])
        self.similarity = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.products.index, index=self.products['product_id']).drop_duplicates()

    def recommend(self, product_id, top_n=5):
        idx = self.indices[product_id]
        sim_scores = list(enumerate(self.similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        product_indices = [i[0] for i in sim_scores]
        return self.products.iloc[product_indices][['product_id', 'product_category_name']]
