from recommendation_config import RecommendationConfig
from algos.collaborative_filtering_based import ItemBasedCFRecommender
from algos.content_based import ContentBasedRecommender
from src.managers.data_manager import DataManager
def train_models():
    config = RecommendationConfig()
    data_manager = DataManager(config=config)
    
    cf_train_df, _, _ = data_manager.load_split_data(save_dir=config.data_dir,
                                                     model_type='cf')
    cbf_train_df, _, _ = data_manager.load_split_data(save_dir=config.data_dir,
                                                      model_type='cbf')

    cf_recommender = ItemBasedCFRecommender(config)
    cbf_recommender = ContentBasedRecommender(config) 

    # Model training
    cf_recommender.fit(ratings_df=cf_train_df)
    cbf_recommender.fit(video_data=cbf_train_df)

if __name__ == "__main__":
    train_models()