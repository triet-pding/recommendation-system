import os
import pickle
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from konlpy.tag import Okt  # Korean language processor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from pathlib import Path
from typing import List, Tuple, Dict, Union
import pandas as pd
import numpy as np
import time
import faiss
import redis
import scipy
from functools import lru_cache
import math
from recommendation_config import RecommendationConfig
from cache_manager import CacheManager

logger = structlog.get_logger()

class ContentBasedRecommender:
    """
    Content-based filtering recommendation system for items with Korean metadata.
    Specifically handles Korean text in 'title' and 'description' attributes.
    """
    
    def __init__(self, config: RecommendationConfig):
        
        # Initialize Korean text processor
        self.okt = Okt() 
        self.korean_stopwords = self._load_korean_stopwords()
        
        # Vector database
        self.index = None
        self.id_mapping = {}
        
        # Initialize transformers
        self.text_vectorizer = None
        self.numerical_scaler = MinMaxScaler()
        self.categorical_encoder = OneHotEncoder(handle_unknown='ignore')

        # Dimension reducer initialize
        self.use_dimensionality_reduction = config.get('dimensionality_reduction', False)
        self.dimension_reducer = None
        self.n_components = config.get('n_components', 100)

        self.model_dir = config.get('model_dir', './models') / "cbf_models"
        self.data_dir = config.get('data_dir', './data')

        # Caching configuration
        self.use_cache = config.get('use_cache', False)
        self.cache_manager = None

        # Configure cache if enabled
        if self.use_cache:
            self.cache_manager = CacheManager(config)
            self.cache_ttl = config.get('cache_ttl', 3600)

        logger.info("Content-based Recommender initialized")
    
    def _configure_cache(self, config: RecommendationConfig, cache_strategy: str) -> None:
        """Configure the cache based on provided configuration"""
        
        if cache_strategy == 'redis':
            try:
                host = config.get('host', 'localhost')
                port = config.get('port', 6379)
                db = config.get('db', 0)
                self.cache_ttl = config.get('ttl', 3600)
                
                self.cache = redis.Redis(host=host, port=port, db=db)
                # Configure LRU eviction policy
                try:
                    self.cache.config_set('maxmemory-policy', 'allkeys-lru')
                except redis.ResponseError:
                    logger.warning("Could not set Redis maxmemory-policy. Make sure Redis is configured correctly.")
                
                logger.info(f"Redis cache configured at {host}:{port}/{db} with TTL={self.cache_ttl}s")
            except ImportError:
                logger.warning("Redis package not installed. Falling back to local cache.")
                self._configure_local_cache(config)
        else:
            self._configure_local_cache(config)

    def _configure_local_cache(self, config: RecommendationConfig) -> None:
        """Configure a simple in-memory LRU cache"""
        max_size = config.get('lru_cache_max_size', 1000)
        
        # Create a dictionary for the cache entries with expiration times
        self.local_cache = {}
        
        # Create an LRU-cached version of the find_similar_videos method
        @lru_cache(maxsize=max_size)
        def cached_find_similar(vid_id: str, top_n: int):
            return self._find_similar_videos_internal(vid_id, top_n)
        
        self.cached_find_similar = cached_find_similar
        logger.info(f"Local LRU cache configured with max_size={max_size}")
        
    def _load_korean_stopwords(self):
        """Load Korean stopwords or use a default set if file not available"""
        logger.info("Load Korean stopwords list")
        try:
            with open(f'{self.data_dir} / korean_stopwords.txt', 'r', encoding='utf-8') as f:
                return set(f.read().splitlines())
        except FileNotFoundError:
            # Default basic Korean stopwords
            return {'이', '그', '저', '것', '수', '등', '들', '및', '에서', '으로', '를', '에', '의', '가', '은', '는', '이런', '저런', '그런'}
    
    def _tokenize_korean_text(self, text):
        """Preprocess Korean text with specialized handling"""
        if not isinstance(text, str):
            return ""
        
        # Normalize text
        text = text.lower()
        
        # Remove special characters but keep Korean, English, numbers
        text = re.sub(r'[^\wㄱ-ㅎㅏ-ㅣ가-힣 ]', ' ', text)
        
        # Tokenize Korean text and select only nouns, adjectives, verbs
        tokens = self.okt.pos(text)
        filtered_tokens = [word for word, pos in tokens if (pos in ['Noun', 'Adjective', 'Verb'] and 
                                                           len(word) > 1 and 
                                                           word not in self.korean_stopwords)]
        
        return ' '.join(filtered_tokens)

    def reduce_dimensions(self, features: Union[np.ndarray, scipy.sparse.spmatrix], 
                          n_components: int = None) -> np.ndarray:
        """Reduce feature dimensionality to save memory and improve performance"""
        if n_components is None:
            n_components = self.n_components
            
        logger.info(f"Reducing dimensions from {features.shape[1]} to {n_components}")
        
        # Check if we should use sparse or dense PCA
        if isinstance(features, scipy.sparse.spmatrix) and features.shape[1] > 1000:
            # For very large sparse matrices, use TruncatedSVD
            if self.dimension_reducer is None:
                self.dimension_reducer = TruncatedSVD(n_components=n_components, random_state=42)
                reduced_features = self.dimension_reducer.fit_transform(features)
            else:
                reduced_features = self.dimension_reducer.transform(features)
        else:
            # For dense matrices or smaller sparse matrices, use PCA
            if isinstance(features, scipy.sparse.spmatrix):
                features = features.toarray()
                
            if self.dimension_reducer is None:
                self.dimension_reducer = PCA(n_components=n_components, random_state=42)
                reduced_features = self.dimension_reducer.fit_transform(features)
            else:
                reduced_features = self.dimension_reducer.transform(features)
        
        logger.info(f"Dimensionality reduction complete. New shape: {reduced_features.shape}")
        return reduced_features

    def preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Korean text columns (title and description)"""
        logger.info(f"Preprocessing text for {len(df)} videos")
        start_time = time.perf_counter()
        # Create copies to avoid modifying the original dataframe
        df_processed = df.copy()
        
        # Tokenize Korean text
        df_processed['title_tokenized'] = df_processed['title'].fillna("").apply(self._tokenize_korean_text)
        df_processed['description_tokenized'] = df_processed['description'].fillna("").apply(self._tokenize_korean_text)
        
        # Combine text features
        df_processed['text_combined'] = df_processed['title_tokenized'] + " " + df_processed['description_tokenized']
        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")
        return df_processed
    
    def extract_text_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract TF-IDF features from preprocessed text"""
        logger.info("Extracting text features")
        start_time = time.perf_counter()
        if self.text_vectorizer is None:
            # Initialize and fit vectorizer if not already done
            self.text_vectorizer = TfidfVectorizer(
                min_df=2, 
                max_df=0.95, 
                max_features=5000, 
                ngram_range=(1, 2), 
                sublinear_tf=True
            )
            text_features = self.text_vectorizer.fit_transform(df['text_combined'])
        else:
            # Use pre-trained vectorizer
            text_features = self.text_vectorizer.transform(df['text_combined'])
        
        # Apply dimensionality reduction if enabled
        if self.use_dimensionality_reduction and text_features.shape[1] > self.n_components:
            text_features = self.reduce_dimensions(text_features, self.n_components)

        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")
        return text_features
    
    def extract_metadata_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and encode numerical and categorical metadata features"""
        logger.info("Extracting metadata features")
        start_time = time.perf_counter()
        # Handle numerical features
        numerical_features = df[['trees_consumed', 'video_duration']].values
        scaled_numerical = self.numerical_scaler.fit_transform(numerical_features)
        
        # Handle categorical features
        categorical_features = df[['purchase_tier', 'pd_category']].values
        encoded_categorical = self.categorical_encoder.fit_transform(categorical_features).toarray()
        
        # Combine all metadata features
        metadata_features = np.hstack((scaled_numerical, encoded_categorical))

        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")
        return metadata_features
    
    def combine_features(self, 
                         text_features: np.ndarray, 
                         metadata_features: np.ndarray, 
                        text_weight: float = 0.7) -> np.ndarray:
        """  
        Args:
            text_features: Sparse or dense text feature matrix
            metadata_features: Dense metadata feature matrix
            text_weight: Weight to apply to text features (0-1)
            
        Returns:
            Combined feature matrix
        """
        logger.info(f"Combining features with text_weight={text_weight}")
        start_time = time.perf_counter()
        
        # Get dimensions
        n_samples = metadata_features.shape[0]
        
        # For metadata features - normalize in-place if possible
        metadata_norm = np.linalg.norm(metadata_features, axis=1, keepdims=True)
        np.divide(metadata_features, np.maximum(metadata_norm, 1e-10), out=metadata_features)
        
        # Apply weight to metadata
        metadata_features *= (1 - text_weight)
        
        # Process text features efficiently based on their type
        if isinstance(text_features, scipy.sparse.spmatrix):
            # For sparse matrices, we handle differently to preserve memory
            # Normalize sparse matrix (this preserves sparsity)
            text_squared = text_features.copy()
            text_squared.data **= 2
            text_norm = np.sqrt(text_squared.sum(axis=1).A1)
            
            # Create a diagonal matrix of normalization factors
            normalizer = scipy.sparse.diags(1.0 / np.maximum(text_norm, 1e-10))
            
            # Normalize and apply weight (still sparse)
            text_normalized = normalizer @ text_features
            text_normalized *= text_weight
            
            # Now we need to combine - convert text to dense only at the final step
            logger.info("Converting sparse text features to dense for final combination")
            combined_features = np.hstack((text_normalized.toarray(), metadata_features))
        else:
            # If already dense, normalize with np operations
            text_norm = np.linalg.norm(text_features, axis=1, keepdims=True)
            text_features /= np.maximum(text_norm, 1e-10)
            text_features *= text_weight
            
            # Combine
            combined_features = np.hstack((text_features, metadata_features))
        
        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds. " 
                f"Combined shape: {combined_features.shape}")
        
        # Report memory usage
        mem_usage = combined_features.nbytes / (1024 * 1024)
        logger.info(f"Memory usage of combined features: {mem_usage:.2f} MB")
        
        return combined_features
    
    def configure_faiss_index(self, feature_dim: int, num_videos: int = None) -> None:
        """Select appropriate FAISS index type based on data size and dimensions"""

        if num_videos is None and self.index is not None:
            num_videos = self.index.ntotal
        elif num_videos is None:
            num_videos = 10000  # Default assumption
            
        logger.info(f"Configuring FAISS index for {num_videos} videos with {feature_dim} dimensions")
        
        if num_videos < 10000:
            # For small datasets, exact search is efficient enough
            self.index = faiss.IndexFlatL2(feature_dim)
            logger.info("Using FlatL2 index for exact search")
        elif num_videos < 100000:
            # For medium datasets, use IVF with 4√n clusters
            n_clusters = int(4 * math.sqrt(num_videos))
            quantizer = faiss.IndexFlatL2(feature_dim)
            self.index = faiss.IndexIVFFlat(quantizer, feature_dim, n_clusters)
            logger.info(f"Using IVFFlat index with {n_clusters} clusters")
        else:
            # For large datasets, use HNSW for better scalability
            self.index = faiss.IndexHNSWFlat(feature_dim, 32)  # 32 neighbors per node
            logger.info("Using HNSWFlat index for large-scale search")
    
    def build_faiss_index(self, feature_matrix: np.ndarray) -> None:
        """Build FAISS index for fast similarity search"""
        logger.info(f"Building FAISS index with {feature_matrix.shape[0]} videos")
        start_time = time.perf_counter()
        # Convert to float32 as required by FAISS
        features_float32 = feature_matrix.astype(np.float32)
        
        # Create and train index
        dimension = features_float32.shape[1]
        if self.index is None:
            self.configure_faiss_index(dimension, feature_matrix.shape[0])
        
        # Some index types need training before adding vectors
        if hasattr(self.index, 'train'):
            logger.info("Training FAISS index")
            self.index.train(features_float32)
        
        # Add vectors to the index
        self.index.add(features_float32)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")

    def fit(self, video_data: pd.DataFrame) -> None:
        """Fit the recommendation model on the provided video data"""
        logger.info(f"Fitting model on {len(video_data)} videos")
        start_time = time.perf_counter()
        # Store original video IDs for mapping
        original_indices = video_data.index.tolist()
        
        # Preprocess text
        processed_df = self.preprocess_text(video_data)
        
        # Extract features
        text_features = self.extract_text_features(processed_df)
        metadata_features = self.extract_metadata_features(processed_df)
        
        # Combine features
        combined_features = self.combine_features(text_features, metadata_features)
        
        # Build search index
        self.build_faiss_index(combined_features)
        
        # Create mapping from FAISS index to original video IDs
        self.id_mapping = {i: original_indices[i] for i in range(len(original_indices))}
        
        # Save models
        self.save_models()
        
        logger.info("Model fitting completed")
        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")

    def find_similar_videos(self, video_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """Find videos similar to the given video ID with optional caching"""
        # Handle caching based on strategy
        if self.use_cache:
            if hasattr(self, 'cache') and self.cache is not None:
                # Redis cache implementation
                cache_key = f"sim_videos:{video_id}:{top_n}"
                cached_result = self.cache.get(cache_key)
                
                if cached_result:
                    self.cache_hits += 1
                    logger.info(f"Cache hit for video_id={video_id}")
                    return pickle.loads(cached_result)
                else:
                    self.cache_misses += 1
            
            elif hasattr(self, 'cached_find_similar'):
                # Local LRU cache implementation
                return self.cached_find_similar(video_id, top_n)
        
        # Cache miss or no cache, perform actual search
        return self._find_similar_videos_internal(video_id, top_n)

    def _find_similar_videos_internal(self, video_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """Find videos similar to the given video ID"""
        
        logger.info(f"Finding {top_n} videos similar to video_id={video_id}")
        
        try:
            # Get the index of the video in our processed data
            video_idx = list(self.id_mapping.values()).index(video_id)
            
            # Get the feature vector for this video
            query_vector = np.array([self.index.reconstruct(video_idx)]).astype(np.float32)
            
            # Search for similar videos
            k = top_n + 1  # +1 because the video itself will be included
            distances, indices = self.index.search(query_vector, k)
            
            # Convert to list of (video_id, similarity_score) tuples
            # Skip the first result (which is the query video itself)
            similar_videos = []
            for i, idx in enumerate(indices[0]):
                if self.id_mapping[idx] != video_id:  # Skip the query video
                    # Convert distance to similarity score (1 / (1 + distance))
                    similarity = 1 / (1 + distances[0][i])
                    similar_videos.append((self.id_mapping[idx], float(similarity)))
                
                if len(similar_videos) == top_n:
                    break
            
            # Cache the result if caching is enabled
            if self.use_cache and hasattr(self, 'cache') and self.cache is not None:
                cache_key = f"sim_videos:{video_id}:{top_n}"
                self.cache.setex(cache_key, self.cache_ttl, pickle.dumps(similar_videos))
            
            return similar_videos
        
        except Exception as e:
            logger.error(f"Error finding similar videos: {str(e)}")
            return []
        
    def save_models(self) -> None:
        """Save trained models and preprocessors to disk"""
        logger.info(f"Saving models to {self.model_dir}")
        start_time = time.perf_counter()
        # Save text vectorizer
        with open(os.path.join(self.model_dir, "text_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.text_vectorizer, f)
        
        # Save numerical scaler
        with open(os.path.join(self.model_dir, "numerical_scaler.pkl"), "wb") as f:
            pickle.dump(self.numerical_scaler, f)
        
        # Save categorical encoder
        with open(os.path.join(self.model_dir, "categorical_encoder.pkl"), "wb") as f:
            pickle.dump(self.categorical_encoder, f)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.model_dir, "faiss_index.bin"))

        if hasattr(self, 'dimension_reducer') and self.dimension_reducer is not None:
            with open(self.model_dir / "dimension_reducer.pkl", "wb") as f:
                pickle.dump(self.dimension_reducer, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save ID mapping
        with open(os.path.join(self.model_dir, "id_mapping.pkl"), "wb") as f:
            pickle.dump(self.id_mapping, f)
        
        end_time = time.perf_counter()
        logger.info("Models saved successfully")
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")

    def load_models(self) -> bool:
        """Load trained models and preprocessors from disk
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        logger.info(f"Loading models from {self.model_dir}")
        
        try:
            # Load text vectorizer
            with open(self.model_dir / "text_vectorizer.pkl", "rb") as f:
                self.text_vectorizer = pickle.load(f)
            
            # Load numerical scaler
            with open(self.model_dir / "numerical_scaler.pkl", "rb") as f:
                self.numerical_scaler = pickle.load(f)
            
            # Load categorical encoder
            with open(self.model_dir / "categorical_encoder.pkl", "rb") as f:
                self.categorical_encoder = pickle.load(f)
            
            # Load dimension reducer if it exists
            dimension_reducer_path = self.model_dir / "dimension_reducer.pkl"
            if dimension_reducer_path.exists():
                with open(dimension_reducer_path, "rb") as f:
                    self.dimension_reducer = pickle.load(f)
                    logger.info("Dimension reducer loaded")
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.model_dir / "faiss_index.bin"))
            
            # Load ID mapping
            with open(self.model_dir / "id_mapping.pkl", "rb") as f:
                self.id_mapping = pickle.load(f)
                
            logger.info("Models loaded successfully")
            return True
        
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {str(e)}")
            return False
        except (pickle.UnpicklingError, ImportError) as e:
            logger.error(f"Error unpickling model files: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading models: {str(e)}")
            return False
        
    def update_model(self, new_video_data: pd.DataFrame) -> None:
        """Add new videos to the existing model without retraining from scratch
        
        Args:
            new_video_data: DataFrame containing new videos to add to the model
            
        Returns:
            bool: True if model was updated successfully, False otherwise
        """
        # Validate required columns
        required_columns = ['title', 'description', 'trees_consumed', 'video_duration', 
                            'purchase_tier', 'pd_category']
        missing_columns = [col for col in required_columns if col not in new_video_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        try:
            # Get current index size to map new IDs
            current_size = self.index.ntotal
            original_indices = new_video_data.index.tolist()
            
            # Process new videos
            processed_df = self.preprocess_text(new_video_data)
            
            # Extract text features
            text_features = self.text_vectorizer.transform(processed_df['text_combined'])
            
            # Handle metadata features with potential new categories
            try:
                # Try with existing encoders
                metadata_features = self.extract_metadata_features(processed_df)
            except (ValueError, KeyError) as e:
                logger.warning(f"Metadata extraction failed with existing encoders: {e}")
                logger.warning("Re-fitting categorical encoder with new data")
                # Re-fit categorical encoder including new data
                # Note: This is a partial solution - for production, consider a more robust approach
                categorical_features = processed_df[['purchase_tier', 'pd_category']].values
                self.categorical_encoder.fit(categorical_features)
                # Try again with updated encoder
                metadata_features = self.extract_metadata_features(processed_df)
            
            # Apply dimensionality reduction if needed
            combined_features = self.combine_features(text_features, metadata_features)
            
            # Add to FAISS index
            combined_features_f32 = combined_features.astype(np.float32)
            self.index.add(combined_features_f32)
            
            # Update ID mapping
            for i, original_id in enumerate(original_indices):
                self.id_mapping[current_size + i] = original_id
                
            # Clear cache for affected videos if using Redis cache
            if self.use_cache and hasattr(self, 'cache') and self.cache is not None:
                # Get all cache keys matching the pattern
                pattern = "sim_videos:*"
                try:
                    keys = self.cache.keys(pattern)
                    if keys:
                        self.cache.delete(*keys)
                        logger.info(f"Cleared {len(keys)} cache entries after model update")
                except Exception as e:
                    logger.warning(f"Could not clear cache: {e}")
            
            # If using local LRU cache, simply clear it
            if hasattr(self, 'cached_find_similar'):
                self.cached_find_similar.cache_clear()
                logger.info("Cleared local LRU cache after model update")
            
            # Save updated models
            self.save_models()
            
            logger.info(f"Model updated with {len(new_video_data)} new videos")
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
    
    def evaluate(self, 
                test_data: pd.DataFrame, user_interactions: pd.DataFrame, 
                rating_threshold: float = 3.0, top_n: int = 10, 
                k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate the recommender system using common metrics.
        
        Args:
            test_data: DataFrame containing test video data with same schema as training data
            user_interactions: DataFrame containing user interaction data with columns 'user_id', 'video_id', 'rating'
                            where 'rating' is a numerical rating value (e.g., 1-5 scale)
            rating_threshold: Minimum rating value to consider an item as relevant/liked (default: 3.0)
            top_n: Number of recommendations to generate for each video
            k_values: List of k values for which to calculate metrics (e.g., precision@k)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating recommender with {len(test_data)} test videos")
        start_time = time.perf_counter()
        
        # Create user-item interaction matrix for ground truth
        # Convert ratings to binary based on threshold
        user_interactions['relevant'] = user_interactions['rating'] >= rating_threshold
        
        user_item_matrix = user_interactions.pivot_table(
            index='user_id', 
            columns='video_id', 
            values='relevant', 
            fill_value=False
        )
        
        # Additionally create a rating matrix for NDCG calculations
        rating_matrix = user_interactions.pivot_table(
            index='user_id',
            columns='video_id',
            values='rating',
            fill_value=0
        )
        
        # Track metrics
        metrics = {}
        metrics['metadata'] = {
            'rating_threshold': rating_threshold,
            'top_n': top_n,
            'k_values': k_values
        }
        
        # Initialize precision and recall metrics for different k values
        for k in k_values:
            metrics[f'precision@{k}'] = 0
            metrics[f'recall@{k}'] = 0
            metrics[f'hit_rate@{k}'] = 0
            metrics[f'ndcg@{k}'] = 0
        
        # Get all available video IDs from the model
        all_video_ids = set(self.id_mapping.values())
        
        # Track which videos were recommended
        recommended_items = set()
        item_pair_distance_sum = 0
        item_pair_count = 0
        
        # Process each user
        user_count = len(user_item_matrix.index)
        for user_idx, user_id in enumerate(user_item_matrix.index):
            if (user_idx + 1) % 100 == 0 or (user_idx + 1) == user_count:
                logger.info(f"Evaluated {user_idx + 1}/{user_count} users")
            
            # Get videos the user has watched
            watched_videos = user_item_matrix.columns[user_item_matrix.loc[user_id]].tolist()
            
            if not watched_videos:
                continue
            
            # For each watched video, generate recommendations and evaluate
            test_videos = set(watched_videos).intersection(all_video_ids)
            
            # Get user's relevant videos (ground truth for testing)
            relevant_videos = set(watched_videos)
            
            # Generate recommendations using "leave-one-out" approach
            all_recommendations = []
            for video_id in test_videos:
                # Find similar videos
                similar_videos = self.find_similar_videos(video_id, top_n=top_n)
                similar_video_ids = [vid_id for vid_id, _ in similar_videos]
                
                # Track recommended items for coverage calculation
                recommended_items.update(similar_video_ids)
                
                # Calculate pairwise distances for diversity
                if len(similar_video_ids) > 1:
                    for i in range(len(similar_video_ids)):
                        idx_i = list(self.id_mapping.values()).index(similar_video_ids[i])
                        vec_i = self.index.reconstruct(idx_i)
                        
                        for j in range(i+1, len(similar_video_ids)):
                            idx_j = list(self.id_mapping.values()).index(similar_video_ids[j])
                            vec_j = self.index.reconstruct(idx_j)
                            
                            # Euclidean distance as a diversity measure
                            distance = np.linalg.norm(vec_i - vec_j)
                            item_pair_distance_sum += distance
                            item_pair_count += 1
                
                # Add to all recommendations for this user
                all_recommendations.extend(similar_video_ids)
            
            # Get unique recommendations
            unique_recommendations = list(dict.fromkeys(all_recommendations))
            
            # Calculate metrics for different k values
            for k in k_values:
                if k > len(unique_recommendations):
                    continue
                    
                # Get top-k recommendations
                top_k_recommendations = unique_recommendations[:k]
                
                # Calculate precision@k: proportion of recommended items that are relevant
                relevant_and_recommended = set(top_k_recommendations).intersection(relevant_videos)
                precision_k = len(relevant_and_recommended) / k if k > 0 else 0
                metrics[f'precision@{k}'] += precision_k
                
                # Calculate recall@k: proportion of relevant items that are recommended
                recall_k = len(relevant_and_recommended) / len(relevant_videos) if relevant_videos else 0
                metrics[f'recall@{k}'] += recall_k
                
                # Hit rate@k: 1 if at least one recommended item is relevant, 0 otherwise
                hit_k = 1 if relevant_and_recommended else 0
                metrics[f'hit_rate@{k}'] += hit_k
                
                # NDCG@k: normalized discounted cumulative gain
                dcg_k = 0
                idcg_k = 0
                
                # Calculate DCG with actual ratings instead of binary relevance
                for i, item_id in enumerate(top_k_recommendations):
                    # Get the rating (or 0 if not rated)
                    rating = rating_matrix.loc[user_id, item_id] if item_id in rating_matrix.columns and user_id in rating_matrix.index else 0
                    if rating > 0:  # Only consider items with ratings
                        dcg_k += rating / np.log2(i + 2)  # +2 because i is 0-indexed
                
                # Calculate ideal DCG (iDCG) using sorted ratings
                # Get all ratings for this user and sort them in descending order
                user_ratings = [rating_matrix.loc[user_id, item] for item in relevant_videos 
                            if item in rating_matrix.columns]
                user_ratings.sort(reverse=True)
                
                for i, rating in enumerate(user_ratings[:k]):
                    idcg_k += rating / np.log2(i + 2)
                
                # Calculate NDCG
                ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0
                metrics[f'ndcg@{k}'] += ndcg_k
        
        # Average metrics across all users
        for k in k_values:
            metrics[f'precision@{k}'] /= user_count
            metrics[f'recall@{k}'] /= user_count
            metrics[f'hit_rate@{k}'] /= user_count
            metrics[f'ndcg@{k}'] /= user_count
        
        # Calculate coverage: percentage of items that were recommended at least once
        metrics['coverage'] = len(recommended_items) / len(all_video_ids) if all_video_ids else 0
        
        # Calculate diversity: average distance between recommended items
        metrics['diversity'] = item_pair_distance_sum / item_pair_count if item_pair_count > 0 else 0
        
        # Calculate novelty: average popularity rank of recommended items
        if recommended_items:
            # Calculate video popularity from average ratings
            video_popularity = user_interactions.groupby('video_id')['rating'].agg(['mean', 'count']).reset_index()
            # Only consider items with a minimum number of ratings
            min_ratings = 5
            popular_videos = video_popularity[video_popularity['count'] >= min_ratings]
            
            if not popular_videos.empty:
                # Normalize to [0,1] range
                max_pop = popular_videos['mean'].max()
                min_pop = popular_videos['mean'].min()
                pop_range = max_pop - min_pop
                
                # Create popularity dictionary
                normalized_popularity = {}
                for _, row in popular_videos.iterrows():
                    if pop_range > 0:
                        normalized_popularity[row['video_id']] = (row['mean'] - min_pop) / pop_range
                    else:
                        normalized_popularity[row['video_id']] = 0.5  # Default if all same popularity
                
                # Calculate average popularity of recommended items (invert for novelty)
                rec_item_popularity = [normalized_popularity.get(vid, 0) for vid in recommended_items 
                                    if vid in normalized_popularity]
                if rec_item_popularity:
                    metrics['novelty'] = 1 - (sum(rec_item_popularity) / len(rec_item_popularity))
                
        
        # Add rating-specific metrics
        metrics['rmse'] = 0
        metrics['mae'] = 0
        
        # Calculate RMSE and MAE for items that have predictions
        prediction_errors = []
        absolute_errors = []
        
        for user_id in user_item_matrix.index:
            # Get this user's actual ratings
            user_ratings = rating_matrix.loc[user_id]
            
            # Only consider videos that are in our test set
            rated_videos = [vid for vid in user_ratings.index if user_ratings[vid] > 0 and vid in all_video_ids]
            
            for video_id in rated_videos:
                # Get actual rating
                actual_rating = rating_matrix.loc[user_id, video_id]
                
                # Get predicted rating based on similar videos
                similar_videos = self.find_similar_videos(video_id, top_n=5)
                
                # If we have no similar videos, skip
                if not similar_videos:
                    continue
                    
                # Calculate weighted average rating using similarity as weights
                weighted_sum = 0
                weight_sum = 0
                
                for sim_vid_id, similarity in similar_videos:
                    # Check if user has rated this similar video
                    if sim_vid_id in rating_matrix.columns and user_id in rating_matrix.index:
                        sim_rating = rating_matrix.loc[user_id, sim_vid_id]
                        if sim_rating > 0:  # Only consider if user has rated
                            weighted_sum += sim_rating * similarity
                            weight_sum += similarity
                
                # Calculate predicted rating if we have weights
                if weight_sum > 0:
                    predicted_rating = weighted_sum / weight_sum
                    
                    # Calculate error
                    error = predicted_rating - actual_rating
                    prediction_errors.append(error ** 2)  # Squared error
                    absolute_errors.append(abs(error))    # Absolute error
        
        # Calculate RMSE and MAE if we have predictions
        if prediction_errors:
            metrics['rmse'] = math.sqrt(sum(prediction_errors) / len(prediction_errors))
            metrics['mae'] = sum(absolute_errors) / len(absolute_errors)
        
        end_time = time.perf_counter()
        logger.info(f"Evaluation completed in {end_time - start_time:.4f} seconds")
        
        # Log metrics
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics