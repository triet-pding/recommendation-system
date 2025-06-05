from src.managers.data_manager import DataManager
from recommendation_config import RecommendationConfig
import structlog

logger = structlog.get_logger()

def prepare_dataset():
    config = RecommendationConfig()
    data_manager = DataManager(config=config)
    
    df = data_manager.load_data()

    logger.info(f"\nPreparing train/test data for Collaborative Filtering model...")
    cf_train_df, cf_test_df = data_manager.temporal_split_recommendation_data(master_df=df, split_strategy='per_user', test_percentage=0.2)

    logger.info(f"\nPreparing train/test data for Content-Based Filtering model...")
    cbf_train_df, cbf_test_df = data_manager.extract_and_split_video_content(interaction_data=df, split_strategy='per_user', test_percentage=0.2)
    
    _ = data_manager.save_split_data(
                                    model_type="cf",
                                    train_df=cf_train_df, 
                                    test_df=cf_test_df, 
                                    save_dir=config.data_dir,
                                    file_format='csv',
                                    compression='gzip'
                                    )
    
    _ = data_manager.save_split_data(
                                    model_type="cbf",
                                    train_df=cbf_train_df, 
                                    test_df=cbf_test_df, 
                                    save_dir=config.data_dir,
                                    file_format='csv',
                                    compression='gzip'
                                    )

    logger.info("--- Data preparing completed ! ---")

if __name__ == "__main__":
    prepare_dataset()