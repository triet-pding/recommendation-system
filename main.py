
from data_manager import DataManager
from recommendation_config import RecommendationConfig
from pathlib import Path
import structlog
import pandas as pd

logger = structlog.get_logger()


# def content_based_recommender_training():
    
#     master_df = pd.read_csv(f"{root_dir}/data/master_df.csv")
   
#     cf_recommender = ContentBasedRecommender(use_cache=False)
#     cf_recommender.fit(video_data=master_df)


# def collaborative_based_recommender_training():

#     master_df = pd.read_csv(f"{root_dir}/data/master_df.csv")
#     cb_recommender = ItemBasedCFRecommender(
#         top_n_similar=10,
#         cache_size=1000,
#         similarity_threshold=0.2
#     )
#     cb_recommender.fit(ratings_df=master_df)
#     cb_recommender.save_model("")

def main():
    config = RecommendationConfig()
    data_manager = DataManager(config=config)
    df = data_manager.load_data()
    train_df, test_df = df

    
main()