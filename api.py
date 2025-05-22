from algos.collaborative_based import ItemBasedCFRecommender
from algos.content_based import ContentBasedRecommender
from recommendation_config import RecommendationConfig

class RecommendationService:
    
    """Handles model training, evaluation, and persistence"""
    def __init__(self, config: RecommendationConfig) -> None:
        self.config = config
        self.cf_recommender = ItemBasedCFRecommender()
        self.cb_recommender = ContentBasedRecommender()