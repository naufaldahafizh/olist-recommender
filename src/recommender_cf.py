# src/recommender_cf.py

from surprise import Dataset, Reader, SVD

class CollaborativeFilteringRecommender:
    def __init__(self, df):
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(df[['customer_id', 'product_id', 'review_score']], reader)
        self.trainset = self.data.build_full_trainset()
        self.model = SVD()
        self.model.fit(self.trainset)

    def get_recommendations(self, user_id_raw, top_n=5):
        user_id_inner = self.trainset.to_inner_uid(user_id_raw)
        items_all = set(self.trainset.all_items())
        items_rated = set([j for (j, _) in self.trainset.ur[user_id_inner]])
        items_unrated = items_all - items_rated

        predictions = []
        for item_inner in items_unrated:
            item_raw = self.trainset.to_raw_iid(item_inner)
            pred = self.model.predict(user_id_raw, item_raw)
            predictions.append((item_raw, pred.est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

