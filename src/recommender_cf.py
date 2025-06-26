from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

class CollaborativeFilteringRecommender:
    def __init__(self, df):
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(df[['customer_id', 'product_id', 'review_score']], reader)
        self.trainset = self.data.build_full_trainset()
        self.model = SVD()
        self.model.fit(self.trainset)

    def recommend_for_user(self, user_id, product_list, top_n=5):
        preds = []
        for product_id in product_list:
            pred = self.model.predict(user_id, product_id)
            preds.append((product_id, pred.est))
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:top_n]
