import pandas as pd
import numpy as np
import heapq
import time
import math
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import List, Tuple, Union, Dict, Optional, Any
import pickle
import structlog
from scipy.sparse import lil_matrix
from recommendation_config import RecommendationConfig
from cache_manager import CacheManager
from pathlib import Path

logger = structlog.get_logger()

class ItemBasedCFRecommender:
    """
    Item-based Collaborative Filtering recommendation system
    specifically designed for video recommendations with user ratings.
    Enhanced with caching and scalability improvements.
    """
    
    def __init__(self, config: RecommendationConfig):
        """
        Initialize the recommender system.
        
        Parameters:
        -----------
        top_n_similar : int, default=10
            Number of most similar items to store for each item
        cache_size : int, default=1024
            Size of the LRU cache for recommendation results
        similarity_threshold : float, default=0.0
            Minimum similarity threshold for considering items related
        """

        self.top_n_similar = config.get('top_n_similar', 10)
        self.similarity_threshold = config.get('similarity_threshold', 0.0)

        # Model state
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        self.popular_items_cache = None
        self.model_metadata = {}


        self.cache_size = config.get('lru_cache_max_size', 1000)
        self.model_dir = config.get('model_dir') / "cf_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cache = config.get('use_cache', False)
        self.cache_manager = None
        self.cache_ttl = config.get('cache_ttl', 3600)

        if self.use_cache:
            try:
                self.cache_manager = CacheManager(config)
                logger.info("Cache manager initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize cache manager: {e}")
                self.use_cache = False
        
        # Performance tracking
        self.fit_time = None
        self.model_stats = {}

    def _validate_rating_data(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the ratings DataFrame."""
        logger.info("Validating ratings data...")
        
        # Check required columns
        required_cols = ['user_id', 'video_id', 'rating']
        missing_cols = [col for col in required_cols if col not in ratings_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing values
        initial_rows = len(ratings_df)
        ratings_df = ratings_df.dropna(subset=required_cols)
        if len(ratings_df) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(ratings_df)} rows with missing values")
        
        # Convert data types
        try:
            ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce')
            ratings_df = ratings_df.dropna(subset=['rating'])
        except Exception as e:
            raise ValueError(f"Error converting ratings to numeric: {e}")
        
    def fit(self, ratings_df: pd.DataFrame, save_model: bool = True):
        """
        Build the item-based collaborative filtering model.
        
        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user_id, video_id, and rating columns
        """
        logger.info("Starting model fitting...")
        start_time = time.time()

        # Clear cache before retraining (important!)
        if hasattr(self, 'cache_manager') and self.cache_manager:
            logger.info("Clearing cache before retraining...")
            self.clear_cache()
        
        try:
            # Validate input data
            ratings_df = self._validate_rating_data(ratings_df)

            if len(ratings_df) == 0:
                raise ValueError("No valid ratings data after filtering")
            
            # Create mappings between original IDs and matrix indices
            self._create_mappings(ratings_df)
            
            # Build the user-item matrix
            self._build_user_item_matrix(ratings_df)
            
            # Calculate item similarity matrix
            self._build_item_similarity_matrix()

            # Pre-compute popular items to use as fallback
            self._compute_popular_items()

            # Store model metadata
            self.fit_time = time.time() - start_time
            self.model_metadata = {
                'n_users': len(self.user_mapping),
                'n_items': len(self.item_mapping),
                'n_ratings': len(ratings_df),
                'sparsity': 1 - (len(ratings_df) / (len(self.user_mapping) * len(self.item_mapping))),
                'fit_time': self.fit_time,
                'similarity_threshold': self.similarity_threshold,
                'top_n_similar': self.top_n_similar
            }
            
            logger.info(f"Model fitting completed in {self.fit_time:.2f} seconds")
            logger.info(f"Model stats: {self.model_metadata}")
            
            # Save model if requested
            if save_model:
                self.save_model()
        
        except Exception as e:
            logger.error(f"Error during model fitting: {e}")
            raise
        
    def _create_mappings(self, ratings_df: pd.DataFrame):
        """Create mappings between original IDs and matrix indices."""
        logger.info("Creating user and item mappings...")
        
        try:
            # Get unique users and items
            unique_users = ratings_df['user_id'].unique()
            unique_items = ratings_df['video_id'].unique()
            
            # Create mappings
            self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
            self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
            
            # Create reverse mappings (index to original ID)
            self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
            self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
            
            logger.info(f"Found {len(unique_users)} unique users and {len(unique_items)} unique videos")
        
        except Exception as e:
            logger.error(f"Error creating mappings: {e}")
            raise
        
    def _build_user_item_matrix(self, ratings_df: pd.DataFrame):
        """Build the user-item matrix from the ratings DataFrame."""
        logger.info("Building user-item matrix...")
        
        try:

            # Convert IDs to matrix indices
            user_indices = [self.user_mapping[user] for user in ratings_df['user_id']]
            item_indices = [self.item_mapping[item] for item in ratings_df['video_id']]
            
            # Create sparse matrix
            n_users = len(self.user_mapping)
            n_items = len(self.item_mapping)
            
            # Convert ratings to float values
            ratings = ratings_df['rating'].values.astype(float)
            
            # Create the sparse matrix
            self.user_item_matrix = csr_matrix((ratings, 
                                                (user_indices, item_indices)), 
                                                shape=(n_users, n_items))
            
            # Eliminate duplicate entries by summing them
            self.user_item_matrix.eliminate_zeros()
            
            logger.info(f"Created user-item matrix of shape {self.user_item_matrix.shape} "
                       f"with {self.user_item_matrix.nnz} non-zero entries")
        
        except Exception as e:
            logger.error(f"Error building user-item matrix: {e}")
            raise
        
    def _build_item_similarity_matrix(self):
        """Calculate item-item similarity matrix with memory-efficient processing."""
        logger.info("Building item similarity matrix...")
        start_time = time.time()
        
        try:
            # Convert to item-user matrix (transpose)
            item_user_matrix = self.user_item_matrix.T.tocsr()
            n_items = item_user_matrix.shape[0]
            
            # Initialize similarity matrix as dictionary for memory efficiency
            self.item_similarity_matrix = {}
            similarity_count = 0
            
            # Process items in batches to reduce memory usage
            for batch_start in range(0, n_items, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_items)
                
                if batch_start % (self.batch_size * 10) == 0:  # Log every 10 batches
                    logger.info(f"Processing similarity batch {batch_start//self.batch_size + 1}/"
                               f"{(n_items + self.batch_size - 1)//self.batch_size}")
                
                try:
                    # Calculate similarities for this batch
                    batch_similarities = self._calculate_batch_similarities(
                        item_user_matrix, batch_start, batch_end
                    )
                    
                    # Store similarities
                    for i, item_idx in enumerate(range(batch_start, batch_end)):
                        similarities = batch_similarities[i]
                        
                        # Filter by threshold and get top N
                        valid_sims = [(idx, sim) for idx, sim in enumerate(similarities) 
                                     if sim > self.similarity_threshold and idx != item_idx]
                        
                        if valid_sims:
                            # Sort and take top N
                            valid_sims.sort(key=lambda x: x[1], reverse=True)
                            top_sims = valid_sims[:self.top_n_similar]
                            
                            # Store as dictionary
                            self.item_similarity_matrix[item_idx] = {
                                idx: sim for idx, sim in top_sims
                            }
                            similarity_count += len(top_sims)
                
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_start}-{batch_end}: {e}")
                    continue
            
            build_time = time.time() - start_time
            logger.info(f"Item similarity matrix built in {build_time:.2f} seconds")
            logger.info(f"Stored {similarity_count} similarity relationships")
            
        except Exception as e:
            logger.error(f"Error building similarity matrix: {e}")
            raise

    def _calculate_batch_similarities(self, item_user_matrix, batch_start: int, batch_end: int) -> np.ndarray:
        """Calculate similarities for a batch of items with error handling."""
        try:
            # Extract batch vectors
            batch_vectors = item_user_matrix[batch_start:batch_end]
            
            # Convert to dense for similarity calculation (only for the batch)
            batch_dense = batch_vectors.toarray()
            
            # Calculate similarities with all items
            similarities = cosine_similarity(batch_dense, item_user_matrix)
            
            return similarities
            
        except MemoryError:
            logger.warning(f"Memory error in batch {batch_start}-{batch_end}, using smaller sub-batches")
            # Fall back to smaller sub-batches
            sub_batch_size = max(1, (batch_end - batch_start) // 4)
            results = []
            
            for sub_start in range(batch_start, batch_end, sub_batch_size):
                sub_end = min(sub_start + sub_batch_size, batch_end)
                sub_vectors = item_user_matrix[sub_start:sub_end].toarray()
                sub_similarities = cosine_similarity(sub_vectors, item_user_matrix)
                results.extend(sub_similarities)
            
            return np.array(results)
        
    def _compute_popular_items(self):
        """Pre-compute popular items using Wilson score with error handling."""
        logger.info("Computing popular items...")
        
        try:
            popular_items = self.recommend_popular_items_wilson(n_recommendations=50)
            self.popular_items_cache = popular_items
            logger.info(f"Cached {len(popular_items)} popular items")
            
        except Exception as e:
            logger.warning(f"Error computing popular items: {e}")
            # Fallback to simple popularity
            try:
                item_counts = np.array(self.user_item_matrix.sum(axis=0))[0]
                top_items = np.argsort(item_counts)[::-1][:50]
                self.popular_items_cache = [
                    (self.reverse_item_mapping[idx], item_counts[idx]) 
                    for idx in top_items if item_counts[idx] > 0
                ]
            except Exception as fallback_error:
                logger.error(f"Fallback popular items computation failed: {fallback_error}")
                self.popular_items_cache = []

    def recommend_for_user(self, user_id: str, n_recommendations: int = 10, 
                          exclude_watched: bool = True) -> List[Tuple[str, float]]:
        """
        Generate personalized recommendations for a user with caching and error handling.
        """
        # Generate cache key
        cache_key = None
        if self.use_cache and self.cache_manager:
            cache_key = self.cache_manager._generate_cache_key(
                "user_rec", user_id, n_recommendations, exclude_watched
            )
            
            # Try to get from cache
            try:
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for user {user_id}")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
        
        try:
            # Generate recommendations
            recommendations = self._recommend_for_user(user_id, n_recommendations, exclude_watched)
            
            # Cache the result
            if self.use_cache and self.cache_manager and cache_key:
                try:
                    self.cache_manager.set(cache_key, recommendations, self.cache_ttl)
                except Exception as e:
                    logger.warning(f"Cache set error: {e}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            # Return popular items as fallback
            return self.popular_items_cache[:n_recommendations] if self.popular_items_cache else []
        
    def _recommend_for_user(self, user_id: str, n_recommendations: int = 10, 
                           exclude_watched: bool = True) -> List[Tuple[str, float]]:
        """Internal method for generating user recommendations."""
        
        # Check if user exists
        if user_id not in self.user_mapping:
            logger.info(f"User {user_id} not found in training data")
            return self.popular_items_cache[:n_recommendations] if self.popular_items_cache else []
        
        user_idx = self.user_mapping[user_id]
        user_vector = self.user_item_matrix[user_idx]
        
        if user_vector.nnz == 0:
            logger.info(f"User {user_id} has no ratings")
            return self.popular_items_cache[:n_recommendations] if self.popular_items_cache else []
        
        # Get user's ratings
        watched_items = user_vector.indices
        watched_ratings = user_vector.data
        
        # Calculate recommendation scores
        scores = defaultdict(float)
        total_similarity = defaultdict(float)
        
        for idx, item_idx in enumerate(watched_items):
            item_rating = watched_ratings[idx]
            
            # Skip very low ratings (configurable threshold)
            if item_rating < 3:
                continue
            
            # Get similar items
            if item_idx in self.item_similarity_matrix:
                for similar_item, similarity in self.item_similarity_matrix[item_idx].items():
                    if exclude_watched and similar_item in watched_items:
                        continue
                    
                    # Weighted score
                    scores[similar_item] += similarity * item_rating
                    total_similarity[similar_item] += similarity
        
        # Normalize scores
        normalized_scores = {
            item_idx: score / total_similarity[item_idx] if total_similarity[item_idx] > 0 else 0
            for item_idx, score in scores.items()
        }
        
        if not normalized_scores:
            return self.popular_items_cache[:n_recommendations] if self.popular_items_cache else []
        
        # Get top recommendations
        top_items = heapq.nlargest(n_recommendations, normalized_scores.keys(), 
                                  key=normalized_scores.get)
        
        recommendations = [
            (self.reverse_item_mapping[item_idx], normalized_scores[item_idx])
            for item_idx in top_items
        ]
        
        return recommendations

    def _wilson_score(self, pos: Union[int, float], n: int, confidence: float = 0.95) -> float:
        """
        Calculate the Wilson score interval for a binomial proportion.
        
        Parameters:
        -----------
        pos : int or float
            Number of positive ratings or sum of ratings
        n : int
            Total number of ratings
        confidence : float, default=0.95
            Confidence level
            
        Returns:
        --------
        float
            Lower bound of Wilson score interval
        """
        if n <= 0:
            return 0.0
        
        # Ensure pos is within valid range
        pos = max(0, min(pos, n))
        phat = pos / n
        
        try:
            z = stats.norm.ppf(1 - (1 - confidence) / 2)
            
            # Wilson score calculation with numerical stability
            denominator = 1 + z*z/n
            if denominator == 0:
                return 0.0
            
            score = (phat + z*z/(2*n) - z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n)) / denominator
            return max(0.0, score)  # Ensure non-negative
            
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logger.warning(f"Wilson score calculation error: {e}")
            return phat  # Fallback to simple proportion

    def recommend_popular_items_wilson(self, n_recommendations: int = 10, 
                                   confidence: float = 0.95, 
                                   normalize_ratings: bool = True) -> List[Tuple[str, float]]:
        """
        Recommend popular items using Wilson score (normalized or unnormalized).
        """
        try:
            n_items = self.user_item_matrix.shape[1]
            wilson_scores = []

            for item_idx in range(n_items):
                col = self.user_item_matrix.getcol(item_idx)
                ratings = col.data

                if len(ratings) == 0:
                    wilson_scores.append(0.0)
                    continue

                if normalize_ratings:
                    norm_ratings = np.clip((ratings - 1) / 4, 0, 1)
                    pos = np.sum(norm_ratings)
                    n = len(norm_ratings)
                else:
                    pos = np.sum(ratings)
                    n = len(ratings) * 5  # max possible sum

                score = self._wilson_score(pos, n, confidence)
                wilson_scores.append(score)

            top_indices = np.argsort(wilson_scores)[::-1][:n_recommendations]
            return [
                (self.reverse_item_mapping[idx], wilson_scores[idx])
                for idx in top_indices if wilson_scores[idx] > 0
            ]

        except Exception as e:
            logger.error(f"Wilson score recommendation failed: {e}")
            return []
    
    def save_model(self, filepath: Optional[str] = None):
        """Save the trained model to disk."""
        if filepath is None:
            filepath = self.model_dir / "cf_model.pkl"
        else:
            filepath = Path(filepath)
        
        try:
            model_data = {
                'user_item_matrix': self.user_item_matrix,
                'item_similarity_matrix': self.item_similarity_matrix,
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_user_mapping': self.reverse_user_mapping,
                'reverse_item_mapping': self.reverse_item_mapping,
                'popular_items_cache': self.popular_items_cache,
                'model_metadata': self.model_metadata,
                'config': {
                    'top_n_similar': self.top_n_similar,
                    'similarity_threshold': self.similarity_threshold,
                    'min_interactions': self.min_interactions
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: Optional[str] = None):
        """Load a trained model from disk."""
        if filepath is None:
            filepath = self.model_dir / "cf_model.pkl"
        else:
            filepath = Path(filepath)
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model state
            self.user_item_matrix = model_data['user_item_matrix']
            self.item_similarity_matrix = model_data['item_similarity_matrix']
            self.user_mapping = model_data['user_mapping']
            self.item_mapping = model_data['item_mapping']
            self.reverse_user_mapping = model_data['reverse_user_mapping']
            self.reverse_item_mapping = model_data['reverse_item_mapping']
            self.popular_items_cache = model_data['popular_items_cache']
            self.model_metadata = model_data.get('model_metadata', {})
            
            # Restore configuration
            config = model_data.get('config', {})
            self.top_n_similar = config.get('top_n_similar', self.top_n_similar)
            self.similarity_threshold = config.get('similarity_threshold', self.similarity_threshold)
            self.min_interactions = config.get('min_interactions', self.min_interactions)
            
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Model metadata: {self.model_metadata}")

            # Clear cache after loading new model
            logger.info("Clearing cache after loading new model...")
            self.clear_cache()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if not self.user_item_matrix:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "metadata": self.model_metadata,
            "cache_stats": {
                "cache_hits": self.cache_manager.cache_hits if self.cache_manager else 0,
                "cache_misses": self.cache_manager.cache_misses if self.cache_manager else 0,
                "cache_enabled": self.use_cache
            }
        }
    
    def clear_cache(self):
        """Clear the recommendation cache."""
        if self.use_cache and self.cache_manager:
            try:
                if self.cache_manager.strategy == 'redis' and self.cache_manager.cache:
                    # Clear Redis cache
                    self.cache_manager.cache.flushdb()
                    logger.info("Redis cache cleared successfully")
                elif self.cache_manager.strategy == 'lru':
                    # Clear LRU cache
                    self.cache_manager.cache.clear()
                    self.cache_manager.access_order.clear()
                    logger.info("LRU cache cleared successfully")
                
                # Reset cache statistics
                self.cache_manager.cache_hits = 0
                self.cache_manager.cache_misses = 0
                
            except Exception as e:
                logger.warning(f"Error clearing cache: {e}")
        else:
            logger.info("No cache to clear (caching disabled or not initialized)")

    
    def update_model(self, new_ratings_df: pd.DataFrame, 
                                      recalculate_similarities: bool = False):
        """
        Update the model with new ratings without complete retraining.
        
        Parameters:
        -----------
        new_ratings_df : pandas.DataFrame
            DataFrame containing new user_id, video_id, and rating columns
        recalculate_similarities : bool, default=False
            Whether to recalculate item similarities (more expensive)
        """
        logger.info("Updating model with new ratings...")
        start_time = time.time()
        
        # Track new users and items
        new_users = set()
        new_items = set()
        
        # Process each new rating
        for _, row in new_ratings_df.iterrows():
            user_id = row['user_id']
            video_id = row['video_id']
            rating = float(row['rating'])
            
            # Handle new users
            if user_id not in self.user_mapping:
                new_users.add(user_id)
                user_idx = len(self.user_mapping)
                self.user_mapping[user_id] = user_idx
                self.reverse_user_mapping[user_idx] = user_id
            else:
                user_idx = self.user_mapping[user_id]
            
            # Handle new items
            if video_id not in self.item_mapping:
                new_items.add(video_id)
                item_idx = len(self.item_mapping)
                self.item_mapping[video_id] = item_idx
                self.reverse_item_mapping[item_idx] = video_id
            else:
                item_idx = self.item_mapping[video_id]
            
            # We need to create a new matrix if dimensions changed
            if new_users or new_items:
                self._rebuild_matrix_with_new_ratings(new_ratings_df)
                break
            
            # Otherwise, update the existing matrix
            # For efficiency, only update the matrix, not recalculate similarities
            # We'd need to convert to LIL format to update efficiently
            lil_matrix = self.user_item_matrix.tolil()
            lil_matrix[user_idx, item_idx] = rating
            self.user_item_matrix = lil_matrix.tocsr()
        
        # Recalculate similarities if requested and we have new items
        if recalculate_similarities and (new_items or new_users):
            self._build_item_similarity_matrix()
        
        # Refresh popular items cache
        self.popular_items_cache = self.recommend_popular_items_wilson()
        
        # Clear recommendation caches
        self.clear_caches()
        
        logger.info(f"Model updated in {time.time() - start_time:.2f} seconds")
        logger.info(f"Added {len(new_users)} new users and {len(new_items)} new items")
    
    def _rebuild_matrix_with_new_ratings(self, new_ratings_df: pd.DataFrame):
        """Rebuild the user-item matrix with the new ratings data"""
        n_users = len(self.user_mapping) 
        n_items = len(self.item_mapping)
        
        # Create a new matrix with the right dimensions
        new_matrix = lil_matrix((n_users, n_items))
        
        # Copy existing data if available
        if self.user_item_matrix is not None:
            # Copy existing values
            old_shape = self.user_item_matrix.shape
            new_matrix[:old_shape[0], :old_shape[1]] = self.user_item_matrix.tolil()
        
        # Add new ratings
        for _, row in new_ratings_df.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['video_id']]
            rating = float(row['rating'])
            new_matrix[user_idx, item_idx] = rating
        
        # Convert back to CSR for efficiency
        self.user_item_matrix = new_matrix.tocsr()
        
        logger.info(f"Rebuilt user-item matrix to shape {self.user_item_matrix.shape}")

    def _predict_rating(self, user_id: str, video_id: str) -> Union[float, None]:
        """
        Predict a user's rating for an item.
        
        Parameters:
        -----------
        user_id : int or str
            User ID
        video_id : int or str
            Video ID
            
        Returns:
        --------
        float or None
            Predicted rating, or None if prediction cannot be made
        """
        # Check if user and item exist in training data
        if user_id not in self.user_mapping or video_id not in self.item_mapping:
            return None
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[video_id]
        
        # Get user's ratings
        user_vector = self.user_item_matrix[user_idx]
        watched_items = user_vector.indices
        watched_ratings = user_vector.data
        
        if len(watched_items) == 0:
            return None
        
        # If user has already rated this item, return the actual rating
        if item_idx in watched_items:
            item_pos = np.where(watched_items == item_idx)[0][0]
            return watched_ratings[item_pos]
        
        # Calculate prediction
        total_weighted_rating = 0
        total_similarity = 0
        
        # Similar items to target item
        if item_idx in self.item_similarity_matrix:
            similar_items = self.item_similarity_matrix[item_idx]
            
            # For each similar item that user has rated
            for rated_item_idx in watched_items:
                if rated_item_idx in similar_items:
                    similarity = similar_items[rated_item_idx]
                    
                    # Get user's rating for this item
                    item_pos = np.where(watched_items == rated_item_idx)[0][0]
                    rating = watched_ratings[item_pos]
                    
                    # Weight by similarity
                    total_weighted_rating += similarity * rating
                    total_similarity += similarity
        
        if total_similarity > 0:
            return total_weighted_rating / total_similarity
        else:
            # Fall back to average rating if no similar items were found
            return np.mean(watched_ratings) if len(watched_ratings) > 0 else None
    
    def _calculate_ndcg(self, recommended_items: List[str], relevant_items: List[str], k: int):
        """Calculate Normalized Discounted Cumulative Gain"""
        # Create relevance vector (1 if item is relevant, 0 otherwise)
        relevance = np.array([1 if item in relevant_items else 0 for item in recommended_items[:k]])
        
        if not relevance.any():
            return 0.0
        
        # Calculate DCG
        # DCG = sum(rel_i / log2(i+1)) for i=1..k
        discounts = np.log2(np.arange(2, len(relevance) + 2))  # log2(i+1) for i=1..k
        dcg = np.sum(relevance / discounts)
        
        # Calculate ideal DCG
        # Sort relevance in descending order for ideal ranking
        ideal_relevance = np.ones(min(len(relevant_items), k))
        if len(ideal_relevance) < k:
            ideal_relevance = np.pad(ideal_relevance, (0, k - len(ideal_relevance)))
            
        ideal_dcg = np.sum(ideal_relevance / discounts[:len(ideal_relevance)])
        
        # Calculate NDCG
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        return ndcg
    
    def _calculate_average_precision(self, recommended_items: List[str], relevant_items: List[str]):
        """Calculate Average Precision for a single user"""
        hits = 0
        sum_precisions = 0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        return sum_precisions / len(relevant_items) if relevant_items else 0

    def evaluate(self, test_data: pd.DataFrame, k: Union[int, List[int]] = 10, metrics: List[str] = None, 
             relevance_threshold: float = 4.0, max_users_for_coverage: int = 100) -> Dict[str, float]:
        """
        Evaluate the recommender system on test data.
        
        Parameters:
        -----------
        test_data : pandas.DataFrame
            DataFrame containing user_id, video_id, and rating columns for testing
        k : Union[int, List[int]], default=10
            Number(s) of recommendations to generate per user for ranking metrics.
            Can be a single integer or list of integers (e.g., [5, 10, 20])
        metrics : List[str], default=None
            List of metrics to compute. If None, computes all available metrics.
            Available metrics: ['rmse', 'mae', 'precision', 'recall', 'ndcg', 'map', 'coverage', 'f1']
        relevance_threshold : float, default=4.0
            Minimum rating to consider an item as relevant for ranking metrics
        max_users_for_coverage : int, default=100
            Maximum number of users to use for coverage calculation (for efficiency)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics with additional metadata
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'precision', 'recall', 'ndcg', 'coverage', 'f1']
        
        # Handle k parameter - convert to list if single integer
        if isinstance(k, int):
            k_values = [k]
        else:
            k_values = sorted(k)  # Sort k values for consistent ordering
        
        max_k = max(k_values)  # Maximum k value for recommendation generation
        
        # Validate inputs
        required_columns = ['user_id', 'video_id', 'rating']
        if not all(col in test_data.columns for col in required_columns):
            raise ValueError(f"Test data must contain columns: {required_columns}")
        
        if test_data.empty:
            raise ValueError("Test data cannot be empty")
        
        logger.info(f"Evaluating recommender on {len(test_data)} test ratings...")
        logger.info(f"Evaluation metrics: {metrics}")
        logger.info(f"K values: {k_values}, Relevance threshold: {relevance_threshold}")
        start_time = time.time()
        
        # Initialize results dictionary
        results = {}
        results['metadata'] = {
            'n_test_ratings': len(test_data),
            'n_test_users': test_data['user_id'].nunique(),
            'n_test_items': test_data['video_id'].nunique(),
            'k_values': k_values,
            'max_k': max_k,
            'relevance_threshold': relevance_threshold
        }
        
        # Get unique users in test data
        test_users = test_data['user_id'].unique()
        
        # Filter users that exist in training data
        valid_test_users = [u for u in test_users if u in self.user_mapping]
        results['metadata']['n_valid_test_users'] = len(valid_test_users)
        
        if not valid_test_users:
            logger.warning("No test users found in training data!")
            return results
        
        logger.info(f"Found {len(valid_test_users)}/{len(test_users)} valid test users")

        # For prediction-based metrics (RMSE, MAE)
        if any(m in metrics for m in ['rmse', 'mae']):
            logger.info("Calculating prediction-based metrics...")
            predictions = []
            actuals = []
            failed_predictions = 0
            
            # Vectorized approach for better performance
            for user_id in valid_test_users:
                user_test_data = test_data[test_data['user_id'] == user_id]
                
                for _, row in user_test_data.iterrows():
                    video_id = row['video_id']
                    actual_rating = row['rating']
                    
                    # Skip items not in training data
                    if video_id not in self.item_mapping:
                        failed_predictions += 1
                        continue
                    
                    # Predict rating
                    try:
                        pred_rating = self._predict_rating(user_id, video_id)
                        if pred_rating is not None:
                            predictions.append(pred_rating)
                            actuals.append(actual_rating)
                        else:
                            failed_predictions += 1
                    except Exception as e:
                        logger.warning(f"Prediction failed for user {user_id}, item {video_id}: {e}")
                        failed_predictions += 1

            results['metadata']['failed_predictions'] = failed_predictions
            results['metadata']['successful_predictions'] = len(predictions)
            
            if predictions:
                # Calculate RMSE
                if 'rmse' in metrics:
                    rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)]))
                    results['rmse'] = rmse
                    
                # Calculate MAE
                if 'mae' in metrics:
                    mae = np.mean([abs(p - a) for p, a in zip(predictions, actuals)])
                    results['mae'] = mae
            else:
                logger.warning("No successful predictions for RMSE/MAE calculation")

        # For ranking-based metrics (Precision, Recall, NDCG, MAP, F1)
        ranking_metrics = ['precision', 'recall', 'ndcg', 'map', 'f1']
        if any(m in metrics for m in ranking_metrics):
            logger.info("Calculating ranking-based metrics...")
            
            # Group test data by user and identify relevant items
            user_test_items = {}
            for user_id in valid_test_users:
                user_test_data = test_data[test_data['user_id'] == user_id]
                relevant_items = set(user_test_data[user_test_data['rating'] >= relevance_threshold]['video_id'])
                if relevant_items:
                    user_test_items[user_id] = relevant_items
            
            results['metadata']['users_with_relevant_items'] = len(user_test_items)
            
            if not user_test_items:
                logger.warning(f"No users have relevant items (rating >= {relevance_threshold})")
            else:
                # Calculate metrics for each user
                user_metrics = {}  # Store metrics per user for each k
                for k_val in k_values:
                    user_metrics[k_val] = {
                        'precisions': [],
                        'recalls': [],
                        'ndcgs': [],
                        'aps': [],
                        'f1_scores': []
                    }
                
                failed_recommendations = 0
                
                for user_id, relevant_items in user_test_items.items():
                    try:
                        # Get top-max_k recommendations (we'll truncate for different k values)
                        user_test_item_ids = set(test_data[test_data['user_id'] == user_id]['video_id'])
                        
                        recs = self._recommend_for_user(
                            user_id, 
                            n_recommendations=max_k, 
                            exclude_watched=True
                        )
                        
                        # Filter out test items to prevent data leakage
                        recs = [(item_id, score) for item_id, score in recs 
                            if item_id not in user_test_item_ids][:max_k]
                        
                        if not recs:
                            failed_recommendations += 1
                            continue
                        
                        # Calculate metrics for each k value
                        for k_val in k_values:
                            # Truncate recommendations to current k
                            rec_items_k = [item_id for item_id, _ in recs[:k_val]]
                            
                            if not rec_items_k:
                                continue
                            
                            # Calculate metrics for this k
                            n_relevant_and_recommended = len(set(rec_items_k) & relevant_items)
                            
                            if 'precision' in metrics:
                                precision = n_relevant_and_recommended / len(rec_items_k)
                                user_metrics[k_val]['precisions'].append(precision)
                            
                            if 'recall' in metrics:
                                recall = n_relevant_and_recommended / len(relevant_items)
                                user_metrics[k_val]['recalls'].append(recall)
                            
                            # F1 Score
                            if 'f1' in metrics:
                                precision_val = n_relevant_and_recommended / len(rec_items_k)
                                recall_val = n_relevant_and_recommended / len(relevant_items)
                                
                                if precision_val + recall_val > 0:
                                    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
                                else:
                                    f1 = 0
                                user_metrics[k_val]['f1_scores'].append(f1)
                            
                            # NDCG
                            if 'ndcg' in metrics:
                                ndcg = self._calculate_ndcg(rec_items_k, relevant_items, k_val)
                                user_metrics[k_val]['ndcgs'].append(ndcg)
                            
                            # Average Precision for MAP
                            if 'map' in metrics:
                                ap = self._calculate_average_precision(rec_items_k, relevant_items)
                                user_metrics[k_val]['aps'].append(ap)
                            
                    except Exception as e:
                        logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                        failed_recommendations += 1
                
                results['metadata']['failed_recommendations'] = failed_recommendations
                
                # Calculate final metrics for each k value
                for k_val in k_values:
                    k_suffix = f"@{k_val}"
                    
                    if 'precision' in metrics and user_metrics[k_val]['precisions']:
                        results[f'precision{k_suffix}'] = np.mean(user_metrics[k_val]['precisions'])
                        results[f'precision{k_suffix}_std'] = np.std(user_metrics[k_val]['precisions'])
                    
                    if 'recall' in metrics and user_metrics[k_val]['recalls']:
                        results[f'recall{k_suffix}'] = np.mean(user_metrics[k_val]['recalls'])
                        results[f'recall{k_suffix}_std'] = np.std(user_metrics[k_val]['recalls'])
                    
                    if 'f1' in metrics and user_metrics[k_val]['f1_scores']:
                        results[f'f1{k_suffix}'] = np.mean(user_metrics[k_val]['f1_scores'])
                        results[f'f1{k_suffix}_std'] = np.std(user_metrics[k_val]['f1_scores'])
                    
                    if 'ndcg' in metrics and user_metrics[k_val]['ndcgs']:
                        results[f'ndcg{k_suffix}'] = np.mean(user_metrics[k_val]['ndcgs'])
                        results[f'ndcg{k_suffix}_std'] = np.std(user_metrics[k_val]['ndcgs'])
                    
                    if 'map' in metrics and user_metrics[k_val]['aps']:
                        results[f'map{k_suffix}'] = np.mean(user_metrics[k_val]['aps'])
                        results[f'map{k_suffix}_std'] = np.std(user_metrics[k_val]['aps'])

        # Calculate catalog coverage
        if 'coverage' in metrics:
            logger.info("Calculating catalog coverage...")
            try:
                all_items = set(self.item_mapping.keys())
                recommended_items = set()
                
                # Use a subset of users for efficiency
                coverage_users = valid_test_users[:max_users_for_coverage]
                results['metadata']['coverage_users_used'] = len(coverage_users)
                
                for user_id in coverage_users:
                    try:
                        recs = self._recommend_for_user(user_id, n_recommendations=k)
                        recommended_items.update([item_id for item_id, _ in recs])
                    except Exception as e:
                        logger.warning(f"Coverage calculation failed for user {user_id}: {e}")
                
                coverage = len(recommended_items) / len(all_items) if all_items else 0
                results['coverage'] = coverage
                results['metadata']['unique_items_recommended'] = len(recommended_items)
                results['metadata']['total_items_in_catalog'] = len(all_items)
                
            except Exception as e:
                logger.error(f"Coverage calculation failed: {e}")
                results['coverage'] = 0
        
        # Calculate additional useful metrics
        results['metadata']['evaluation_time_seconds'] = time.time() - start_time
        
        logger.info(f"Evaluation completed in {results['metadata']['evaluation_time_seconds']:.2f} seconds")
        
        # Log results with better formatting
        logger.info("=== Evaluation Results ===")
        for metric, value in results.items():
            if metric != 'metadata':
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.4f}")
                else:
                    logger.info(f"{metric}: {value}")
        
        # Log metadata
        logger.info("=== Metadata ===")
        for key, value in results['metadata'].items():
            logger.info(f"{key}: {value}")
                
        return results