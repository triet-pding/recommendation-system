import pandas as pd
import numpy as np
import heapq
import time
import math
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import List, Tuple, Union, Dict
from functools import lru_cache
import pickle
from scipy import sparse
import structlog
from scipy.sparse import lil_matrix
from recommendation_config import RecommendationConfig

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

        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        self.popular_items_cache = None

        self.cache_size = config.get('lru_cache_max_size', 1000)
        self.model_dir = config.get('model_dir') / "cf_models"
        
        # Configure the LRU cache decorators
        self._configure_caches()
        
    def _configure_caches(self):
        """Configure LRU caches with the specified size"""
        # Make the recommend_for_user method use LRU cache
        self.recommend_for_user = lru_cache(maxsize=self.cache_size)(self._recommend_for_user)
        # Make the recommend_similar_items method use LRU cache
        self.recommend_similar_items = lru_cache(maxsize=self.cache_size)(self._recommend_similar_items)
        
    def fit(self, ratings_df: pd.DataFrame):
        """
        Build the item-based collaborative filtering model.
        
        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user_id, video_id, and rating columns
        """
        logger.info("Starting model fitting...")
        start_time = time.time()
        
        # Create mappings between original IDs and matrix indices
        self._create_mappings(ratings_df)
        
        # Build the user-item matrix
        self._build_user_item_matrix(ratings_df)
        
        # Calculate item similarity matrix
        self._build_item_similarity_matrix()
        
        # Pre-compute popular items to use as fallback
        self.popular_items_cache = self.recommend_popular_items_wilson()
        
        logger.info(f"Model fitting completed in {time.time() - start_time:.2f} seconds")
        
    def _create_mappings(self, ratings_df: pd.DataFrame):
        """Create mappings between original IDs and matrix indices."""
        logger.info("Creating user and item mappings...")
        
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
        
    def _build_user_item_matrix(self, ratings_df: pd.DataFrame):
        """Build the user-item matrix from the ratings DataFrame."""
        logger.info("Building user-item matrix...")
        
        # Convert IDs to matrix indices
        user_indices = [self.user_mapping[user] for user in ratings_df['user_id']]
        item_indices = [self.item_mapping[item] for item in ratings_df['video_id']]
        
        # Create sparse matrix
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        
        # Convert ratings to float values
        ratings = ratings_df['rating'].values.astype(float)
        
        # Create the sparse matrix
        self.user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), 
                                          shape=(n_users, n_items))
        
        logger.info(f"Created user-item matrix of shape {self.user_item_matrix.shape}")
        
    def _build_item_similarity_matrix(self, batch_size: int = 1000):
        """Calculate the item-item similarity matrix using cosine similarity."""
        logger.info("Building item similarity matrix...")
        start_time = time.time()
        
        # Convert to item-user matrix (transpose)
        item_user_matrix = self.user_item_matrix.T.tocsr()
        
        # Calculate similarity matrix
        n_items = item_user_matrix.shape[0]
        self.item_similarity_matrix = {}
        
        # Process items in batches to reduce memory usage
        
        for batch_start in range(0, n_items, batch_size):
            batch_end = min(batch_start + batch_size, n_items)
            if batch_start > 0:
                logger.info(f"Processing batch {batch_start//batch_size + 1}/{(n_items + batch_size - 1)//batch_size}...")
            
            # Extract the batch of item vectors
            batch_vectors = item_user_matrix[batch_start:batch_end].toarray()
            
            # Calculate similarities with all items at once for this batch
            similarities = cosine_similarity(batch_vectors, item_user_matrix.toarray())
            
            # Process each item in the batch
            for i, item_idx in enumerate(range(batch_start, batch_end)):
                # Set self-similarity to -1 to exclude it
                similarities[i, item_idx] = -1
                
                # Filter by threshold before finding top N
                valid_indices = np.where(similarities[i] > self.similarity_threshold)[0]
                
                # Get top N similar items
                if len(valid_indices) > self.top_n_similar:
                    # Use argpartition for efficiency (O(n) instead of O(n log n) for full sort)
                    top_indices = np.argpartition(similarities[i, valid_indices], -self.top_n_similar)[-self.top_n_similar:]
                    top_similar_indices = valid_indices[top_indices]
                else:
                    top_similar_indices = valid_indices
                
                # Store only nonzero similarities in a dictionary (sparse representation)
                self.item_similarity_matrix[item_idx] = {
                    sim_idx: similarities[i, sim_idx] 
                    for sim_idx in top_similar_indices 
                    if similarities[i, sim_idx] > 0
                }
        
        logger.info(f"Item similarity matrix built in {time.time() - start_time:.2f} seconds")
        
        # Calculate memory usage
        similarity_size = sum(len(similarities) for similarities in self.item_similarity_matrix.values())
        logger.info(f"Item similarity matrix contains {similarity_size} nonzero elements")

    def _wilson_score(self, pos: Union[int, float], n: int, confidence: float = 0.95):
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
        if n == 0:
            return 0
        
        # Handle case where pos > n (which can happen when using sum of ratings)
        if pos > n:
            # For Wilson score calculation, proportion must be between 0 and 1
            # Normalize the positive score to be within valid range
            phat = 1.0
        else:
            phat = pos / n
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        # Ensure phat is in the valid range [0, 1]
        phat = min(max(phat, 0), 1)
        
        # Wilson score calculation
        score = (phat + z*z/(2*n) - z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n)) / (1 + z*z/n)
        
        return score

    def recommend_popular_items_wilson(self, n_recommendations: int = 10, confidence: float = 0.95, normalize_ratings: bool = True):
        """
        Recommend most popular items based on Wilson score.
        Used as fallback for cold-start users.
        
        Parameters:
        -----------
        n_recommendations : int, default=10
            Number of recommendations to generate
        confidence : float, default=0.95
            Confidence level for Wilson score
        normalize_ratings : bool, default=True
            Whether to normalize ratings to 0-1 range
                
        Returns:
        --------
        list of tuples
            List of (video_id, wilson_score) tuples
        """
        # Convert user-item matrix to array for easier processing
        # Use CSR matrix methods to avoid full conversion to dense array
        ratings_sum = np.array(self.user_item_matrix.sum(axis=0))[0]
        
        # Count nonzero elements per column (number of ratings per item)
        ratings_count = np.array(self.user_item_matrix.getnnz(axis=0))
        
        # Prepare to store results
        wilson_scores = []
        
        # Process each item
        for item_idx in range(self.user_item_matrix.shape[1]):
            # Skip items with no ratings
            if ratings_count[item_idx] == 0:
                wilson_scores.append(0)
                continue
            
            if normalize_ratings:
                # For normalized ratings, we need to fetch the actual ratings
                # Get the column for this item
                col = self.user_item_matrix.getcol(item_idx)
                # Extract nonzero values
                ratings = col.data
                
                # Normalize ratings to 0-1 range (assuming 1-5 scale)
                # Ensure we handle ratings outside the expected range
                min_rating = 1  # Minimum expected rating
                max_rating = 5  # Maximum expected rating
                norm_ratings = np.clip((ratings - min_rating) / (max_rating - min_rating), 0, 1)
                pos = np.sum(norm_ratings)
                # Use number of ratings as n for normalized case
                n = len(ratings)
            else:
                # Use sum of ratings as "positive" outcome
                pos = ratings_sum[item_idx]
                # For non-normalized case, we need a reasonable denominator
                # We'll use the maximum possible rating sum (n * max_rating)
                n = ratings_count[item_idx] * 5  # Assuming 5 is the max rating
            
            # Calculate Wilson score
            score = self._wilson_score(pos, n, confidence)
            wilson_scores.append(score)
        
        # Use numpy for efficient top-N selection
        if n_recommendations >= len(wilson_scores):
            top_indices = np.argsort(wilson_scores)[::-1]
        else:
            # Use argpartition for more efficient selection
            top_indices = np.argpartition(wilson_scores, -n_recommendations)[-n_recommendations:]
            # Sort the top N
            top_indices = top_indices[np.argsort([-wilson_scores[i] for i in top_indices])]
        
        # Convert to original video IDs
        recommendations = [
            (self.reverse_item_mapping[idx], wilson_scores[idx])
            for idx in top_indices
        ]
        
        return recommendations
    
    def _recommend_for_user(self, user_id: str, n_recommendations: int = 10, exclude_watched: bool = True) -> List[Tuple]:
        """
        Generate personalized recommendations for a user.
        This internal method does the actual work and is wrapped by the cached public method.
        
        Parameters:
        -----------
        user_id : str
            Original user ID
        n_recommendations : int, default=10
            Number of recommendations to generate
        exclude_watched : bool, default=True
            Whether to exclude videos the user has already watched
            
        Returns:
        --------
        list of tuples
            List of (video_id, predicted_rating) tuples
        """
        # Check if user exists in training data
        if user_id not in self.user_mapping:
            logger.info(f"User {user_id} not found in training data. Using popular items instead.")
            return self.popular_items_cache[:n_recommendations]
        
        # Get user index
        user_idx = self.user_mapping[user_id]
        
        # Get user's ratings efficiently from sparse matrix
        user_vector = self.user_item_matrix[user_idx]
        watched_items = user_vector.indices
        watched_ratings = user_vector.data
        
        if len(watched_items) == 0:
            logger.info(f"User {user_id} has no ratings. Using popular items instead.")
            return self.popular_items_cache[:n_recommendations]
        
        # Initialize recommendation scores as sparse dictionary for efficiency
        scores = defaultdict(float)
        total_similarity = defaultdict(float)
        
        # For each rated item
        for idx, item_idx in enumerate(watched_items):
            # Get user's rating for this item
            item_rating = watched_ratings[idx]
            
            # Skip low ratings (optional - you might want to consider negative feedback)
            if item_rating < 3:
                continue
                
            # Get similar items
            if item_idx in self.item_similarity_matrix:
                # For each similar item
                for similar_item, similarity in self.item_similarity_matrix[item_idx].items():
                    # Skip if user has already watched this item and we want to exclude watched
                    if exclude_watched and similar_item in watched_items:
                        continue
                    
                    # Weight by both rating and similarity
                    scores[similar_item] += similarity * item_rating
                    # Track total similarity for normalization
                    total_similarity[similar_item] += similarity
        
        # Normalize scores by total similarity for more stable predictions
        normalized_scores = {
            item_idx: score/total_similarity[item_idx] if total_similarity[item_idx] > 0 else score
            for item_idx, score in scores.items()
        }
        
        # If we have no recommendations after filtering
        if len(normalized_scores) == 0:
            return self.popular_items_cache[:n_recommendations]
        
        # Sort by score and take top N
        top_item_indices = heapq.nlargest(n_recommendations, 
                                         normalized_scores.keys(), 
                                         key=normalized_scores.get)
        
        # Convert back to original video IDs
        recommendations = [
            (self.reverse_item_mapping[item_idx], normalized_scores[item_idx])
            for item_idx in top_item_indices
        ]
        
        return recommendations
    
    def _recommend_similar_items(self, video_id: str, n_recommendations=10) -> List:
        """
        Find videos similar to a given video.
        This internal method does the actual work and is wrapped by the cached public method.
        
        Parameters:
        -----------
        video_id : str
            Original video ID
        n_recommendations : int, default=10
            Number of similar videos to recommend
            
        Returns:
        --------
        list of tuples
            List of (video_id, similarity_score) tuples
        """
        # Check if item exists in training data
        if video_id not in self.item_mapping:
            logger.info(f"Video {video_id} not found in training data.")
            return []
        
        # Get item index
        item_idx = self.item_mapping[video_id]
        
        # If no similarity data for this item
        if item_idx not in self.item_similarity_matrix:
            logger.info(f"No similarity data for video {video_id}.")
            return []
        
        # Get similar items
        similar_items = self.item_similarity_matrix[item_idx]
        
        # Sort by similarity and take top N
        top_similar = heapq.nlargest(n_recommendations, 
                                    similar_items.keys(), 
                                    key=similar_items.get)
        
        # Convert back to original video IDs
        recommendations = [
            (self.reverse_item_mapping[sim_idx], similar_items[sim_idx])
            for sim_idx in top_similar
        ]
        
        return recommendations
    
    def clear_caches(self):
        """Clear all LRU caches to free memory or refresh recommendations"""
        self.recommend_for_user.cache_clear()
        self.recommend_similar_items.cache_clear()
    
    def save_model(self, filepath: str, model_file_name:str):
        """Save the model to disk"""
        
        # Create a dictionary with the model data
        model_data = {
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'item_similarity_matrix': self.item_similarity_matrix,
            'popular_items_cache': self.popular_items_cache,
            'top_n_similar': self.top_n_similar,
            'similarity_threshold': self.similarity_threshold,
            'cache_size': self.cache_size
        }
        
        # Save sparse matrix separately for efficiency
        if self.user_item_matrix is not None:
            
            sparse.save_npz(f"{filepath}/{model_file_name}_user_item_matrix.npz", self.user_item_matrix)
            
        # Save the rest of the data
        with open(f"{filepath}/{model_file_name}", 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved model from disk"""
        import pickle
        from scipy import sparse
        
        # Load the model data
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            top_n_similar=model_data['top_n_similar'],
            cache_size=model_data['cache_size'],
            similarity_threshold=model_data['similarity_threshold']
        )
        
        # Load the attributes
        instance.user_mapping = model_data['user_mapping']
        instance.item_mapping = model_data['item_mapping']
        instance.reverse_user_mapping = model_data['reverse_user_mapping']
        instance.reverse_item_mapping = model_data['reverse_item_mapping']
        instance.item_similarity_matrix = model_data['item_similarity_matrix']
        instance.popular_items_cache = model_data['popular_items_cache']
        
        # Load sparse matrix if it exists
        try:
            instance.user_item_matrix = sparse.load_npz(f"{filepath}_user_item_matrix.npz")
        except FileNotFoundError:
            logger.info("User-item matrix file not found, loading only the similarity data")
        
        return instance
    
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