import pickle
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from konlpy.tag import Okt  # Korean language processor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from pathlib import Path
from typing import List, Tuple, Dict, Union, Set, Any
import pandas as pd
import numpy as np
import time
import faiss
import gc
import scipy
import math
from recommendation_config import RecommendationConfig
from src.managers.cache_manager import CacheManager
import psutil

logger = structlog.get_logger()

class ContentBasedRecommender:
    """
    Content-based filtering recommendation system for items with Korean metadata.
    Specifically handles Korean text in 'title' and 'description' attributes.
    """
    
    def __init__(self, config: RecommendationConfig):

        self.model_dir = Path(config.model_dir) / "cbf_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = Path(config.get('data_dir', './data'))
        
        # Initialize Korean text processor
        self.okt = Okt() 
        self.korean_stopwords = self._load_korean_stopwords()
        
        # Vector database
        self.index = None
        self.feature_matrix = None
        self.id_mapping = {} # faiss id -> original id
        self.reverse_id_mapping = {} #  original video id -> faiss id
        
        # Initialize transformers
        self.text_vectorizer = None
        self.numerical_scaler = MinMaxScaler()
        self.categorical_encoder = OneHotEncoder(handle_unknown='ignore')

        # Dimension reducer initialize
        self.use_dimensionality_reduction = config.get('dimensionality_reduction', False)
        self.dimension_reducer = None
        self.n_components = config.get('n_components', 100)
        self.max_components = min(config.get('n_components', 100), 500)  # Cap at 500
        self.batch_size = 1000  # Configurable batch size

        # Caching configuration
        self.use_cache = config.get('use_cache', False)
        self.cache_manager = None

        # Configure cache if enabled
        if self.use_cache:
            self.cache_manager = CacheManager(config)
            self.cache_ttl = config.get('cache_ttl', 3600)

        logger.info(f"--- Content-based Recommender initialized ---\n")

        
    def _load_korean_stopwords(self):
        """Load Korean stopwords or use a default set if file not available"""
        logger.info("Load Korean stopwords list")
        try:
            stopwords_file = self.data_dir / 'korean_stopwords.txt'
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                return set(f.read().splitlines())
        except FileNotFoundError:
            # Default basic Korean stopwords
            return {'이', '그', '저', '것', '수', '등', '들', '및', '에서', '으로', '를', '에', '의', '가', '은', '는', '이런', '저런', '그런'}
    
    def _tokenize_korean_text(self, text: str) -> str:
        """Preprocess Korean text with specialized handling"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            # Normalize text
            text = text.lower().strip()
            
            # Remove special characters but keep Korean, English, numbers
            text = re.sub(r'[^\wㄱ-ㅎㅏ-ㅣ가-힣 ]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Tokenize Korean text and select only nouns, adjectives, verbs
            tokens = self.okt.pos(text)
            filtered_tokens = [
                word for word, pos in tokens 
                if (pos in ['Noun', 'Adjective', 'Verb'] and 
                    len(word) > 1 and 
                    word not in self.korean_stopwords)
            ]
            
            return ' '.join(filtered_tokens)
        
        except Exception as e:
            logger.error(f"Error tokenizing Korean text: {e}")
            return ""

    def preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Korean text columns (title and description)"""
        logger.info(f"Preprocessing text for {len(df)} videos...")
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
        logger.info("Extracting text features...")
        start_time = time.perf_counter()
        if self.text_vectorizer is None:
            # Initialize and fit vectorizer if not already done
            self.text_vectorizer = TfidfVectorizer(
                min_df=2, 
                max_df=0.95, 
                max_features=5000, 
                ngram_range=(1, 2), 
                sublinear_tf=True,
                lowercase=True
            )
            text_features = self.text_vectorizer.fit_transform(df['text_combined'])
        else:
            # Use pre-trained vectorizer
            text_features = self.text_vectorizer.transform(df['text_combined'])
        
        # Apply dimensionality reduction if enabled
        # if self.use_dimensionality_reduction and text_features.shape[1] > self.n_components:
        #     text_features = self.reduce_dimensions(text_features, self.n_components)

        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")
        return text_features
    
    def extract_metadata_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and encode numerical and categorical metadata features"""
        logger.info("Extracting metadata features...")
        start_time = time.perf_counter()

        # Handle numerical features with proper error handling
        numerical_cols = ['trees_consumed', 'video_duration']
        missing_cols = [col for col in numerical_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing numerical columns: {missing_cols}")
            # Fill missing columns with zeros
            for col in missing_cols:
                df[col] = 0
        
        numerical_features = df[numerical_cols].fillna(0).values
        
        # Fit scaler if not already fitted
        if not hasattr(self.numerical_scaler, 'scale_'):
            scaled_numerical = self.numerical_scaler.fit_transform(numerical_features)
        else:
            scaled_numerical = self.numerical_scaler.transform(numerical_features)
        
        # Handle categorical features with proper error handling
        categorical_cols = ['purchase_tier', 'pd_category']
        missing_cols = [col for col in categorical_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing categorical columns: {missing_cols}")
            # Fill missing columns with 'unknown'
            for col in missing_cols:
                df[col] = ''
        
        categorical_features = df[categorical_cols].fillna('').values
        
        # Fit encoder if not already fitted
        if not hasattr(self.categorical_encoder, 'categories_'):
            encoded_categorical = self.categorical_encoder.fit_transform(categorical_features).toarray()
        else:
            encoded_categorical = self.categorical_encoder.transform(categorical_features).toarray()
        
        # Combine all metadata features
        metadata_features = np.hstack((scaled_numerical, encoded_categorical))

        end_time = time.perf_counter()
        logger.info(f"Metadata feature extraction completed in {end_time - start_time:.4f} seconds")
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
        logger.info(f"Combining features with text_weight={text_weight}...")
        start_time = time.perf_counter()

        # Validate inputs
        if text_features.shape[0] != metadata_features.shape[0]:
            raise ValueError(f"Feature matrices have different number of samples: "
                           f"{text_features.shape[0]} vs {metadata_features.shape[0]}")
        
        # Ensure text_weight is valid
        text_weight = max(0.0, min(1.0, text_weight))
        metadata_weight = 1.0 - text_weight
        
        # Process text features efficiently based on their type
        if scipy.sparse.issparse(text_features):
            # For sparse matrices, normalize while preserving sparsity
            text_squared = text_features.copy()
            text_squared.data **= 2
            text_norm = np.sqrt(text_squared.sum(axis=1).A1)
            
            # Create normalizer with safe division
            normalizer = scipy.sparse.diags(1.0 / np.maximum(text_norm, 1e-10))
            
            # Normalize and apply weight
            text_normalized = normalizer @ text_features
            text_normalized *= text_weight
            
            # Convert to dense for final combination
            text_features_final = text_normalized.toarray()
        else:
            # Dense text features
            text_norm = np.linalg.norm(text_features, axis=1, keepdims=True)
            text_features_final = text_features / np.maximum(text_norm, 1e-10)
            text_features_final *= text_weight
        
        # Process metadata features
        metadata_norm = np.linalg.norm(metadata_features, axis=1, keepdims=True)
        metadata_normalized = metadata_features / np.maximum(metadata_norm, 1e-10)
        metadata_normalized *= metadata_weight
        
        # Combine features
        combined_features = np.hstack((text_features_final, metadata_normalized))
        
        end_time = time.perf_counter()
        logger.info(f"Feature combination completed in {end_time - start_time:.4f} seconds. "
                   f"Combined shape: {combined_features.shape}")
        
        # Report memory usage
        mem_usage = combined_features.nbytes / (1024 * 1024)
        logger.info(f"Memory usage of combined features: {mem_usage:.2f} MB")
        
        return combined_features
    
    def configure_faiss_index(self, feature_dim: int, num_videos: int = None) -> None:
        """Select appropriate FAISS index type based on data size and dimensions"""

        if num_videos is None:
            num_videos = 10000  # Default assumption
            
        logger.info(f"Configuring FAISS index for {num_videos} videos with {feature_dim} dimensions...")
        
        if num_videos < 10000:
            # For small datasets, exact search is efficient
            self.index = faiss.IndexFlatL2(feature_dim)
            logger.info("Using FlatL2 index for exact search")
        elif num_videos < 100000:
            # For medium datasets, use IVF with optimized cluster count
            n_clusters = min(int(4 * math.sqrt(num_videos)), num_videos // 10)
            n_clusters = max(n_clusters, 1)  # Ensure at least 1 cluster
            
            quantizer = faiss.IndexFlatL2(feature_dim)
            self.index = faiss.IndexIVFFlat(quantizer, feature_dim, n_clusters)
            logger.info(f"Using IVFFlat index with {n_clusters} clusters")
        else:
            # For large datasets, use HNSW for better scalability
            self.index = faiss.IndexHNSWFlat(feature_dim, 32)  # 32 neighbors per node
            self.index.hnsw.efConstruction = 200  # Build-time search depth
            self.index.hnsw.efSearch = 128       # Query-time search depth
            logger.info("Using HNSWFlat index for large-scale search")

    
    def build_faiss_index(self, feature_matrix: np.ndarray) -> None:
        """Build FAISS index for fast similarity search"""
        logger.info(f"Building FAISS index with {feature_matrix.shape[0]} videos...")
        start_time = time.perf_counter()

        # Validate input
        if feature_matrix.size == 0:
            raise ValueError("Feature matrix is empty")
        
        # Convert to float32 as required by FAISS
        features_float32 = feature_matrix.astype(np.float32)
        
        # Create and train index
        dimension = features_float32.shape[1]
        if self.index is None:
            self.configure_faiss_index(dimension, feature_matrix.shape[0])
        
        # Some index types need training before adding vectors
        if hasattr(self.index, 'train') and hasattr(self.index, 'is_trained'):
            if not self.index.is_trained:
                logger.info("Training FAISS index")
                self.index.train(features_float32)
        
        # Add vectors to the index
        self.index.add(features_float32)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")


    def fit(self, video_data: pd.DataFrame) -> None:
        """Fit the recommendation model on the provided video data"""
        logger.info("Starting content-based model fitting...")
        logger.info(f"Fitting model on {len(video_data)} videos...")
        start_time = time.perf_counter()

        if len(video_data) == 0:
            raise ValueError("Video data is empty")
        
        # Store unique original video IDs for mapping 
        original_indices = video_data['video_id'].tolist()
        
        # Preprocess text
        processed_df = self.preprocess_text(video_data)
        
        # Extract features
        text_features = self.extract_text_features(processed_df)
        metadata_features = self.extract_metadata_features(processed_df)
        
        # Combine features
        combined_features = self.combine_features(text_features, metadata_features)

        self.feature_matrix = combined_features.astype(np.float32)
        
        # Build search index
        self.build_faiss_index(combined_features)
        
        # Create bi-directional mapping from FAISS index to original video IDs and vice versa
        self.id_mapping = {i: original_indices[i] for i in range(len(original_indices))}
        self.reverse_id_mapping = {original_indices[i]: i for i in range(len(original_indices))}

        logger.info(f"Created ID mappings for {len(self.id_mapping)} videos")
        logger.info(f"Sample original video IDs: {list(self.reverse_id_mapping.keys())[:5]}")
        
        # Save models
        self.save_models()
        
        logger.info("Model fitting completed")
        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")

    def find_similar_videos(self, video_id, top_n: int = 10) -> List[Tuple[str, float]]:
        """Find videos similar to the given video ID with optional caching"""
        # Validate inputs
        if top_n <= 0:
            logger.warning(f"Invalid top_n value: {top_n}, using default of 10")
            top_n = 10
        
        # Handle caching based on strategy
        if self.use_cache and self.cache_manager:
            cache_key = self.cache_manager._generate_cache_key(
                "sim_videos", video_id, top_n
            )
            
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for video_id={video_id}, top_n={top_n}")
                return cached_result
        
        # Cache miss or no cache, perform actual search
        result = self._find_similar_videos_internal(video_id, top_n)
        
        # Store result in cache if caching is enabled
        if self.use_cache and self.cache_manager and result:
            cache_key = self.cache_manager._generate_cache_key(
                "sim_videos", video_id, top_n
            )
            self.cache_manager.set(cache_key, result, self.cache_ttl)
        
        return result
    
    def _normalize_video_id(self, video_id: str):
        # """Normalize video_id to match the type used in training data"""
    
        # Try exact match first
        if video_id in self.reverse_id_mapping:
            return video_id
        
        # Get sample of training data to determine common type
        sample_ids = list(self.reverse_id_mapping.keys())[:20]
        if not sample_ids:
            return None
        
        # Determine the most common type
        type_counts = {}
        for vid_id in sample_ids:
            vid_type = type(vid_id).__name__
            type_counts[vid_type] = type_counts.get(vid_type, 0) + 1
        
        most_common_type = max(type_counts, key=type_counts.get)
        
        # Try different conversion strategies
        conversion_attempts = []
        
        if most_common_type == 'int':
            try:
                if isinstance(video_id, str):
                    if '.' in video_id:
                        conversion_attempts.append(int(float(video_id)))
                    else:
                        conversion_attempts.append(int(video_id))
                else:
                    conversion_attempts.append(int(video_id))
            except (ValueError, TypeError):
                pass
        
        elif most_common_type == 'str':
            conversion_attempts.append(str(video_id))
        
        elif most_common_type in ['float', 'float64']:
            try:
                conversion_attempts.append(float(video_id))
            except (ValueError, TypeError):
                pass
        
        # Also try the original type
        conversion_attempts.append(video_id)
        
        # Test each conversion
        for attempt in conversion_attempts:
            if attempt in self.reverse_id_mapping:
                if attempt != video_id:
                    logger.debug(f"Normalized {video_id} ({type(video_id).__name__}) -> {attempt} ({type(attempt).__name__})")
                return attempt
        
        return None

    def _find_similar_videos_internal(self, video_id: str, top_n: int = 10) -> List[Tuple[int, float]]:
        """FIXED: Find videos similar to the given video ID with robust similarity search"""
        
        # logger.debug(f"Finding {top_n} videos similar to video_id={video_id} (type: {type(video_id)})")

        if self.index is None:
            logger.error("Model not fitted. Call fit() first.")
            return []
        
        if not hasattr(self, 'feature_matrix') or self.feature_matrix is None:
            logger.error("Feature matrix not available. Model may not be properly fitted.")
            return []
        
        try:
            # Normalize video ID to match training data types
            normalized_video_id = self._normalize_video_id(video_id)
            
            if normalized_video_id is None:
                logger.warning(f"Video ID {video_id} not found after normalization")
                logger.debug(f"Available video ID types: {set(type(vid).__name__ for vid in list(self.reverse_id_mapping.keys())[:20])}")
                logger.debug(f"Sample available video IDs: {list(self.reverse_id_mapping.keys())[:10]}")
                return []
            
            # Get the FAISS index for this video
            video_idx = self.reverse_id_mapping[normalized_video_id]
            # logger.debug(f"Found video at FAISS index: {video_idx}")
            
            # FIXED: Use stored feature matrix instead of reconstruct()
            query_vector = self.feature_matrix[video_idx:video_idx+1].astype(np.float32)
            
            if query_vector.size == 0:
                logger.error(f"Empty query vector for video_idx {video_idx}")
                return []
            
            # Search for similar videos
            k = min(top_n + 1, self.index.ntotal)  # +1 because the video itself will be included
            distances, indices = self.index.search(query_vector, k)
            
            # logger.debug(f"FAISS search returned {len(indices[0])} results")
            # logger.debug(f"Distances: {distances[0][:5]}")  # Log first 5 distances
            # logger.debug(f"Indices: {indices[0][:5]}")      # Log first 5 indices
            
            # Convert to list of (video_id, similarity_score) tuples
            similar_videos = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    # logger.debug(f"Skipping invalid index at position {i}")
                    continue
                    
                if idx >= len(self.id_mapping):
                    logger.warning(f"Index {idx} out of range for id_mapping (size: {len(self.id_mapping)})")
                    continue
                
                current_video_id = self.id_mapping[idx]
                
                # Skip the query video itself
                if current_video_id == normalized_video_id:
                    # logger.debug(f"Skipping query video itself: {current_video_id}")
                    continue
                
                # FIXED: Better similarity calculation
                distance = float(distances[0][i])
                if distance < 0:
                    logger.warning(f"Negative distance {distance} for video {current_video_id}")
                    continue
                    
                # Convert distance to similarity score
                # Using exponential decay: similarity = exp(-distance)
                similarity = np.exp(-distance)
                
                similar_videos.append((current_video_id, similarity))
                # logger.debug(f"Added similar video: {current_video_id} (similarity: {similarity:.4f})")
                
                if len(similar_videos) >= top_n:
                    break
            
            # Sort by similarity score (descending)
            similar_videos.sort(key=lambda x: x[1], reverse=True)
            
            return similar_videos
            
        except KeyError as e:
            logger.error(f"KeyError finding similar videos for video_id={video_id}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error finding similar videos for video_id={video_id}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
        
    def save_models(self) -> bool:
        """Save trained models and preprocessors to disk"""
        logger.info(f"Saving models to {self.model_dir}...")
        start_time = time.perf_counter()
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(self.model_dir / "faiss_index.bin"))
            
            # Save other components
            components = {
                'text_vectorizer': self.text_vectorizer,
                'numerical_scaler': self.numerical_scaler,
                'categorical_encoder': self.categorical_encoder,
                'dimension_reducer': self.dimension_reducer,
                'id_mapping': self.id_mapping,
                'reverse_id_mapping': self.reverse_id_mapping,
                'korean_stopwords': self.korean_stopwords,
                'feature_matrix': self.feature_matrix
            }
            
            with open(self.model_dir / "components.pkl", 'wb') as f:
                pickle.dump(components, f)
            
            logger.info(f"Models saved to {self.model_dir}")
            logger.info(f"Total execution time: {time.perf_counter() - start_time:.4f} seconds.")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
        

    def load_models(self) -> None:
        """Load trained models and preprocessors from disk
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            # Load FAISS index
            index_path = self.model_dir / "faiss_index.bin"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            
            # Load other components
            components_path = self.model_dir / "components.pkl"
            if components_path.exists():
                with open(components_path, 'rb') as f:
                    components = pickle.load(f)
                
                self.text_vectorizer = components.get('text_vectorizer')
                self.numerical_scaler = components.get('numerical_scaler')
                self.categorical_encoder = components.get('categorical_encoder')
                self.dimension_reducer = components.get('dimension_reducer')
                self.id_mapping = components.get('id_mapping', {})
                self.reverse_id_mapping = components.get('reverse_id_mapping', {})
                self.korean_stopwords = components.get('korean_stopwords', set())
                self.feature_matrix = components.get('feature_matrix')

                if not self.reverse_id_mapping and self.id_mapping:
                    self.reverse_id_mapping = {v: k for k, v in self.id_mapping.items()}
                    logger.info("Created reverse ID mapping from existing ID mapping")
            
            logger.info(f"--- Content-based models loaded from {self.model_dir}")
            logger.info(f"Loaded {len(self.id_mapping)} video mappings")
        
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {str(e)}")
            raise
        except (pickle.UnpicklingError, ImportError) as e:
            logger.error(f"Error unpickling model files: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading models: {str(e)}")
            raise
        
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
                faiss_idx = current_size + i
                self.id_mapping[faiss_idx] = original_id
                self.reverse_id_mapping[original_id] = faiss_idx
                
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

    def get_trained_video_ids(self) -> Set:
        """
        UTILITY FUNCTION: Get all original video IDs that the model was trained on
        
        Returns:
            set: Set of original video IDs available in the trained model
        """
        return set(self.reverse_id_mapping.keys())
    
    def get_faiss_index_for_video(self, video_id: int) -> int:
        """
        UTILITY FUNCTION: Get FAISS index for a given original video ID
        
        Args:
            video_id: Original video ID
            
        Returns:
            int: FAISS index, or -1 if video not found
        """
        normalized_id = self._normalize_video_id(video_id)
        if normalized_id is not None and normalized_id in self.reverse_id_mapping:
            return self.reverse_id_mapping[normalized_id]
        
        return -1
        #return self.reverse_id_mapping.get(video_id, -1)
    
    def get_original_id_for_faiss_index(self, faiss_idx: int) -> int:
        """
        UTILITY FUNCTION: Get original video ID for a given FAISS index
        
        Args:
            faiss_idx: FAISS index
            
        Returns:
            int: Original video ID, or -1 if index not found
        """
        return self.id_mapping.get(faiss_idx, -1)

    
    def evaluate(self, 
                user_interactions: pd.DataFrame, 
                rating_threshold: float = 3.0, top_n: int = 10, 
                k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate the recommender system using common metrics.
        
        Args:
            : Testing proportion DataFrame containing user interaction data with columns 'user_id', 'video_id', 'rating'
                            where 'rating' is a numerical rating value (e.g., 1-5 scale)
            rating_threshold: Minimum rating value to consider an item as relevant/liked (default: 3.0)
            top_n: Number of recommendations to generate for each video
            k_values: List of k values for which to calculate metrics (e.g., precision@k)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating content-based recommender with {len(user_interactions)} test videos...")

        start_time = time.perf_counter()
        
        # Store rating_threshold as instance variable for access in other methods
        self.rating_threshold = rating_threshold
        
        # Input validation
        if user_interactions.empty:
            raise ValueError("Testing data is empty")
        
        required_cols = ['user_id', 'video_id', 'rating']
        missing_cols = [col for col in required_cols if col not in user_interactions.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in testing dataframe: {missing_cols}")
        
        # Remove any rows with missing values
        user_interactions = user_interactions.dropna(subset=required_cols)

        # DEBUG: Check rating distribution
        logger.debug(f"Rating distribution:")
        logger.debug(f"Min: {user_interactions['rating'].min()}, Max: {user_interactions['rating'].max()}")
        logger.debug(f"Mean: {user_interactions['rating'].mean():.2f}, Median: {user_interactions['rating'].median():.2f}")
        logger.debug(f"Ratings >= {rating_threshold}: {(user_interactions['rating'] >= rating_threshold).sum()}/{len(user_interactions)}")

        # Check video ID overlap
        test_video_ids = set(user_interactions['video_id'].unique())
        trained_video_ids = self.get_trained_video_ids()
        
        overlap = test_video_ids.intersection(trained_video_ids)
        
        logger.info(f"Test dataset contains {len(test_video_ids)} unique video IDs")
        logger.info(f"Model was trained on {len(trained_video_ids)} video IDs")
        logger.info(f"Overlap between test and trained video IDs: {len(overlap)}")
        
        if len(overlap) == 0:
            logger.error("NO OVERLAP between test video IDs and trained video IDs!")
            raise ValueError("No overlap between test and training video IDs. Cannot evaluate.")
        
        if len(overlap) < len(test_video_ids) * 0.1:
            logger.warning(f"Low overlap ({len(overlap)}/{len(test_video_ids)} = {len(overlap)/len(test_video_ids)*100:.1f}%) between test and training video IDs")
        
        # Filter user interactions to only include videos in the trained model
        user_interactions_filtered = user_interactions[
            user_interactions['video_id'].isin(trained_video_ids)
        ]
        
        logger.info(f"Filtered test data: {len(user_interactions_filtered)} interactions (from {len(user_interactions)})")

        if len(user_interactions_filtered) == 0:
            raise ValueError("No valid interactions after filtering for trained video IDs")
        
        # Create user-item matrix with actual rating values
        user_item_matrix = user_interactions_filtered.pivot_table(
            index='user_id', 
            columns='video_id', 
            values='rating',
            fill_value=0
        )

        # DEBUG: Check user-item matrix
        logger.info(f"User-item matrix shape: {user_item_matrix.shape}")
        
        # Pre-compute video popularity for novelty calculation
        video_popularity = self._compute_video_popularity(user_interactions_filtered)
        
        # Initialize metrics tracking
        metrics = self._initialize_metrics(rating_threshold, top_n, k_values)
        
        # Pre-compute similarity cache to avoid redundant calculations
        similarity_cache = {}
        
        # Track metrics across all users
        user_metrics = {f'precision@{k}': [] for k in k_values}
        user_metrics.update({f'recall@{k}': [] for k in k_values})
        user_metrics.update({f'hit_rate@{k}': [] for k in k_values})
        user_metrics.update({f'ndcg@{k}': [] for k in k_values})
        
        recommended_items = set()
        diversity_distances = []
        prediction_errors = []
        absolute_errors = []
        
        # DEBUG: Track evaluation progress
        users_with_recommendations = 0
        users_with_relevant_items = 0
        total_recommendations = 0
        
        # Process each user
        user_count = len(user_item_matrix.index)
        for user_idx, user_id in enumerate(user_item_matrix.index):
            if (user_idx + 1) % 100 == 0 or (user_idx + 1) == user_count:
                logger.info(f"Evaluated {user_idx + 1}/{user_count} users")
            
            try:
                user_results = self._evaluate_user(
                    user_id, user_item_matrix, trained_video_ids, 
                    similarity_cache, k_values, top_n
                )
                
                # DEBUG: Track user results
                if user_results['recommended_items']:
                    users_with_recommendations += 1
                    total_recommendations += len(user_results['recommended_items'])
                
                # Check if user has any relevant items
                user_row = user_item_matrix.loc[user_id]
                relevant_videos = set(user_row[user_row >= self.rating_threshold].index.tolist())
                if relevant_videos:
                    users_with_relevant_items += 1
                
                # Accumulate user-level metrics
                for metric_name, value in user_results['metrics'].items():
                    if metric_name in user_metrics:
                        user_metrics[metric_name].append(value)
                
                # Accumulate other metrics
                recommended_items.update(user_results['recommended_items'])
                diversity_distances.extend(user_results['diversity_distances'])
                prediction_errors.extend(user_results['prediction_errors'])
                absolute_errors.extend(user_results['absolute_errors'])
                
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        # DEBUG: Log evaluation statistics
        logger.info(f"Users with recommendations: {users_with_recommendations}/{user_count}")
        logger.info(f"Users with relevant items: {users_with_relevant_items}/{user_count}")
        logger.info(f"Average recommendations per user: {total_recommendations/max(users_with_recommendations, 1):.2f}")
        
        # Log metric list lengths for debugging
        for metric_name, values in user_metrics.items():
            logger.info(f"{metric_name}: {len(values)} values, avg: {np.mean(values) if values else 0:.4f}")
        
        # Calculate final metrics
        self._finalize_metrics(
            metrics, user_metrics, recommended_items, trained_video_ids,
            diversity_distances, prediction_errors, absolute_errors, video_popularity
        )
        
        end_time = time.perf_counter()
        logger.info(f"Evaluation completed in {end_time - start_time:.4f} seconds")
        
        # Log key metrics
        key_metrics = ['precision@10', 'recall@10', 'ndcg@10', 'coverage', 'diversity']
        for metric in key_metrics:
            if metric in metrics:
                logger.info(f"{metric}: {metrics[metric]:.4f}")
        
        return metrics


    def _compute_video_popularity(self, user_interactions: pd.DataFrame, min_ratings: int = 5) -> Dict:
        """Pre-compute video popularity metrics for novelty calculation."""
        video_stats = user_interactions.groupby('video_id')['rating'].agg(['mean', 'count']).reset_index()
        popular_videos = video_stats[video_stats['count'] >= min_ratings]
        
        if popular_videos.empty:
            return {}
        
        # Normalize popularity to [0,1] range
        max_pop = popular_videos['mean'].max()
        min_pop = popular_videos['mean'].min()
        pop_range = max_pop - min_pop
        
        normalized_popularity = {}
        for _, row in popular_videos.iterrows():
            if pop_range > 0:
                normalized_popularity[row['video_id']] = (row['mean'] - min_pop) / pop_range
            else:
                normalized_popularity[row['video_id']] = 0.5
        
        return normalized_popularity

    def _initialize_metrics(self, rating_threshold: float, top_n: int, k_values: List[int]) -> Dict:
        """Initialize the metrics dictionary."""
        metrics = {
            'metadata': {
                'rating_threshold': rating_threshold,
                'top_n': top_n,
                'k_values': k_values
            }
        }
        
        # Initialize all metrics to 0
        metric_names = ['precision', 'recall', 'hit_rate', 'ndcg']
        for metric in metric_names:
            for k in k_values:
                metrics[f'{metric}@{k}'] = 0.0
        
        metrics.update({
            'coverage': 0.0,
            'diversity': 0.0,
            'novelty': 0.0,
            'rmse': 0.0,
            'mae': 0.0
        })
        
        return metrics
    

    def _evaluate_user(self, user_id: str, user_item_matrix: pd.DataFrame, 
                       all_video_ids: Set, similarity_cache: dict, 
                       k_values: List[int], top_n: int) -> Dict[str, str]:
        """Evaluate metrics for a single user."""
        
        if user_id not in user_item_matrix.index:
            return self._empty_user_result(k_values)
        
        user_row = user_item_matrix.loc[user_id]

        watched_videos = user_row[user_row > 0].index.tolist()
        
        if not watched_videos:
            return self._empty_user_result(k_values)
        
        # Filter to videos available in our model
        watched_videos = list(set(watched_videos).intersection(all_video_ids))
        if not watched_videos:
            return self._empty_user_result(k_values)
        
        # Get relevant videos (liked videos with rating >= threshold)
        relevant_videos = set(user_row[user_row >= self.rating_threshold].index.tolist())
        relevant_videos = relevant_videos.intersection(all_video_ids)
        
        if not relevant_videos:
            logger.warning(f"No relevant videos from user {user_id}")
            return self._empty_user_result(k_values)
        
        # Use more seed videos for better recommendations
        seed_videos = watched_videos[:min(20, len(watched_videos))] 
        
        # Generate recommendations with better deduplication
        recommendation_scores = {}  # video_id -> max_similarity_score
        diversity_distances = []
        
        for video_id in seed_videos:
            if video_id not in similarity_cache:
                similarity_cache[video_id] = self.find_similar_videos(video_id, top_n=top_n*2)  # Get more candidates
            
            similar_videos = similarity_cache[video_id]
            
            for sim_vid_id, similarity_score in similar_videos:
                # Skip if already watched
                if sim_vid_id in watched_videos:
                    continue
                    
                # Keep the highest similarity score for each video
                if sim_vid_id not in recommendation_scores:
                    recommendation_scores[sim_vid_id] = similarity_score
                else:
                    recommendation_scores[sim_vid_id] = max(
                        recommendation_scores[sim_vid_id], similarity_score
                    )
            
            # Calculate diversity for this seed's recommendations
            similar_video_ids = [vid_id for vid_id, _ in similar_videos[:top_n]]
            if len(similar_video_ids) > 1:
                distances = self._calculate_pairwise_distances(similar_video_ids)
                diversity_distances.extend(distances)
        
        # Sort recommendations by similarity score and take top-k
        sorted_recommendations = sorted(
            recommendation_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        unique_recommendations = [vid_id for vid_id, _ in sorted_recommendations]
      
        # Ensure we have recommendations
        if not unique_recommendations:
            logger.warning(f"No recommendations generated for user {user_id}")
            return self._empty_user_result(k_values)
        
        # Calculate user-level metrics
        user_metrics = {}
        for k in k_values:
            if k <= len(unique_recommendations):
              
                top_k = unique_recommendations[:k]
                user_metrics.update(self._calculate_user_k_metrics(
                    top_k, relevant_videos, user_item_matrix, user_id, k
                ))
            else:
                # If we have fewer recommendations than k, use all available
                if unique_recommendations:
              
                    user_metrics.update(self._calculate_user_k_metrics(
                        unique_recommendations, relevant_videos, user_item_matrix, user_id, k
                    ))
                else:
                 
                    user_metrics.update({
                        f'precision@{k}': 0.0,
                        f'recall@{k}': 0.0,
                        f'hit_rate@{k}': 0.0,
                        f'ndcg@{k}': 0.0
                    })
        
        # Calculate prediction errors
        pred_errors, abs_errors = self._calculate_prediction_errors(
            user_id, user_item_matrix, all_video_ids, similarity_cache
        )
        
        return {
            'metrics': user_metrics,
            'recommended_items': set(unique_recommendations),
            'diversity_distances': diversity_distances,
            'prediction_errors': pred_errors,
            'absolute_errors': abs_errors
        }

    def _empty_user_result(self, k_values: List[int]) -> Dict:
        """Return empty result structure for users with no valid data."""
        empty_metrics = {}
        for k in k_values:
            empty_metrics.update({
                f'precision@{k}': 0.0,
                f'recall@{k}': 0.0,
                f'hit_rate@{k}': 0.0,
                f'ndcg@{k}': 0.0
            })
        
        return {
            'metrics': empty_metrics,
            'recommended_items': set(),
            'diversity_distances': [],
            'prediction_errors': [],
            'absolute_errors': []
        }

    def _calculate_pairwise_distances(self, video_ids: List) -> List[float]:
        """Calculate pairwise distances between videos efficiently."""
        distances = []
        
        # Pre-compute vectors for all videos
        vectors = {}
        for vid_id in video_ids:
            try:
                if vid_id in self.reverse_id_mapping:
                    faiss_idx = self.reverse_id_mapping[vid_id]
                    vectors[vid_id] = self.index.reconstruct(faiss_idx)
            except (ValueError, IndexError, RuntimeError) as e:
                logger.warning(f"Could not get vector for video {vid_id}: {e}")
                continue
        
        # Calculate pairwise distances
        vid_list = list(vectors.keys())
        for i in range(len(vid_list)):
            for j in range(i + 1, len(vid_list)):
                try:
                    distance = np.linalg.norm(vectors[vid_list[i]] - vectors[vid_list[j]])
                    distances.append(distance)
                except Exception as e:
                    logger.warning(f"Error calculating distance: {e}")
                    continue
        
        return distances

    def _calculate_user_k_metrics(self, top_k_recs: List, relevant_videos: set, 
                                  user_item_matrix: pd.DataFrame, user_id: str, k: int) -> Dict:
        """Calculate precision, recall, hit rate, and NDCG for a user at k."""

        normalized_recs = []
        for rec in top_k_recs:
            norm_rec = self._normalize_video_id(rec)
            if norm_rec is not None and norm_rec in user_item_matrix.columns:
                normalized_recs.append(norm_rec)
        
        # Normalize relevant videos as well
        normalized_relevant = set()
        for rel_vid in relevant_videos:
            norm_rel = self._normalize_video_id(rel_vid)
            if norm_rel is not None:
                normalized_relevant.add(norm_rel)
        
        # Find intersection
        relevant_and_recommended = set(normalized_recs).intersection(normalized_relevant)
        
        # Calculate metrics
        # Precision@k
        precision_k = len(relevant_and_recommended) / len(normalized_recs) if normalized_recs else 0.0
        
        # Recall@k  
        recall_k = len(relevant_and_recommended) / len(normalized_relevant) if normalized_relevant else 0.0
        
        # Hit rate@k
        hit_k = 1.0 if relevant_and_recommended else 0.0
        
        # NDCG@k
        ndcg_k = self._calculate_ndcg(normalized_recs, user_item_matrix, user_id, k)
        
        return {
            f'precision@{k}': precision_k,
            f'recall@{k}': recall_k,
            f'hit_rate@{k}': hit_k,
            f'ndcg@{k}': ndcg_k
        }

    def _calculate_ndcg(self, recommendations: List, user_item_matrix: pd.DataFrame, 
                        user_id: str, k: int) -> float:
        """Calculate NDCG@k for a user."""
        if user_id not in user_item_matrix.index:
            return 0.0
        
        user_row = user_item_matrix.loc[user_id]
        
        # Calculate DCG
        dcg = 0.0
        for i, item_id in enumerate(recommendations[:k]):
            # Normalize item ID
            norm_item_id = self._normalize_video_id(item_id)
            if norm_item_id is not None and norm_item_id in user_item_matrix.columns:
                rating = user_row[norm_item_id]
                if rating > 0:
                    dcg += rating / np.log2(i + 2)
        
        # Calculate IDCG (Ideal DCG)
        user_ratings = user_row[user_row > 0].sort_values(ascending=False)
        
        idcg = 0.0
        for i, rating in enumerate(user_ratings[:k]):
            idcg += rating / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0



    def _calculate_prediction_errors(self, user_id: str, user_item_matrix: pd.DataFrame, 
                                   all_video_ids: set, similarity_cache: dict) -> Tuple[List[float], List[float]]:
        """Calculate prediction errors for a user."""
        if user_id not in user_item_matrix.index:
            return [], []
        
        user_row = user_item_matrix.loc[user_id]
        rated_videos = [vid for vid in user_row.index 
                        if user_row[vid] > 0 and vid in all_video_ids]
        
        pred_errors = []
        abs_errors = []
        
        for video_id in rated_videos:
            actual_rating = user_row[video_id]
            
            # Get cached similarities
            if video_id not in similarity_cache:
                continue
            
            similar_videos = similarity_cache[video_id]
            
            # Calculate weighted prediction
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for sim_vid_id, similarity in similar_videos:
                if sim_vid_id in user_item_matrix.columns:
                    sim_rating = user_row[sim_vid_id]
                    if sim_rating > 0:
                        weighted_sum += sim_rating * similarity
                        weight_sum += similarity
            
            # Calculate error if we have a prediction
            if weight_sum > 0:
                predicted_rating = weighted_sum / weight_sum
                error = predicted_rating - actual_rating
                pred_errors.append(error ** 2)
                abs_errors.append(abs(error))
        
        return pred_errors, abs_errors

    def _finalize_metrics(self, metrics: Dict, user_metrics: Dict, recommended_items: set,
                          all_video_ids: set, diversity_distances: List, 
                          prediction_errors: List, absolute_errors: List, video_popularity: Dict):
        """Calculate final averaged metrics."""
        
        # Average user-level metrics with fallback to 0
        for metric_name, values in user_metrics.items():
            if values:  # Only calculate if we have values
                metrics[metric_name] = np.mean(values)
            else:
                # Explicitly set to 0 if no values (this was missing before)
                metrics[metric_name] = 0.0
                logger.warning(f"No values found for {metric_name}")
        
        # Coverage
        metrics['coverage'] = len(recommended_items) / len(all_video_ids) if all_video_ids else 0.0
        
        # Diversity
        metrics['diversity'] = np.mean(diversity_distances) if diversity_distances else 0.0
        
        # Novelty
        if recommended_items and video_popularity:
            rec_popularities = [video_popularity.get(vid, 0) for vid in recommended_items 
                                if vid in video_popularity]
            if rec_popularities:
                metrics['novelty'] = 1.0 - np.mean(rec_popularities)
            else:
                metrics['novelty'] = 0.0
        else:
            metrics['novelty'] = 0.0
        
        # RMSE and MAE
        if prediction_errors:
            metrics['rmse'] = math.sqrt(np.mean(prediction_errors))
            metrics['mae'] = np.mean(absolute_errors)
        else:
            metrics['rmse'] = 0.0
            metrics['mae'] = 0.0

    
    # def reduce_dimensions(self, features: Union[np.ndarray, scipy.sparse.spmatrix], 
    #                   n_components: int = None) -> np.ndarray:
    #     """Reduce feature dimensionality with memory-safe approach"""
    #     if n_components is None:
    #         n_components = self.n_components
            
    #     # Don't reduce if already smaller than target
    #     if features.shape[1] <= n_components:
    #         logger.info(f"Features already have {features.shape[1]} dimensions, skipping reduction")
    #         return features.toarray() if scipy.sparse.issparse(features) else features
        
    #     # Log memory usage before processing
    #     memory_before = psutil.virtual_memory().percent
    #     logger.info(f"Memory usage before reduction: {memory_before:.1f}%")
    #     logger.info(f"Reducing dimensions from {features.shape[1]} to {n_components}")
        
    #     # Calculate safe n_components
    #     max_components = min(
    #         n_components, 
    #         features.shape[0] - 1, 
    #         features.shape[1] - 1,
    #         1000  # Cap at 1000 to prevent memory issues
    #     )
        
    #     if max_components <= 0:
    #         logger.warning("Cannot reduce dimensions, returning original features")
    #         return features.toarray() if scipy.sparse.issparse(features) else features
        
    #     try:
    #         # Force garbage collection before processing
    #         gc.collect()
            
    #         # Check available memory
    #         available_memory_gb = psutil.virtual_memory().available / (1024**3)
    #         logger.info(f"Available memory: {available_memory_gb:.2f} GB")
            
    #         if available_memory_gb < 1.0:  # Less than 1GB available
    #             logger.warning("Low memory available, skipping dimensionality reduction")
    #             return features.toarray() if scipy.sparse.issparse(features) else features
            
    #         # Memory-safe processing based on matrix type and size
    #         if scipy.sparse.issparse(features):
    #             return self._reduce_sparse_safely(features, max_components)
    #         else:
    #             return self._reduce_dense_safely(features, max_components)
                
    #     except Exception as e:
    #         logger.error(f"Error during dimensionality reduction: {e}")
    #         logger.info("Returning original features due to error")
    #         # Clean up and return original
    #         gc.collect()
    #         return features.toarray() if scipy.sparse.issparse(features) else features

    # def _reduce_sparse_safely(self, features: scipy.sparse.spmatrix, n_components: int) -> np.ndarray:
    #     """Safely reduce sparse matrix dimensions"""
    #     logger.info("Processing sparse matrix with TruncatedSVD")
        
    #     try:
    #         # Use incremental approach for very large matrices
    #         if features.shape[0] > 10000 or features.shape[1] > 5000:
    #             logger.info("Using batch processing for large sparse matrix")
    #             return self._batch_reduce_sparse(features, n_components)
            
    #         # Standard TruncatedSVD for smaller matrices
    #         if self.dimension_reducer is None or not isinstance(self.dimension_reducer, TruncatedSVD):
    #             self.dimension_reducer = TruncatedSVD(
    #                 n_components=n_components,
    #                 algorithm='randomized',
    #                 n_iter=5,  # Reduce iterations to save memory
    #                 random_state=42
    #             )
                
    #         # Convert to CSR format if not already (more efficient for SVD)
    #         if not scipy.sparse.isspmatrix_csr(features):
    #             features = features.tocsr()
                
    #         reduced_features = self.dimension_reducer.fit_transform(features)
            
    #         # Force cleanup
    #         del features
    #         gc.collect()
            
    #         logger.info(f"Sparse reduction complete. New shape: {reduced_features.shape}")
    #         return reduced_features
            
    #     except Exception as e:
    #         logger.error(f"Error in sparse reduction: {e}")
    #         raise

    # def _reduce_dense_safely(self, features: np.ndarray, n_components: int) -> np.ndarray:
    #     """Safely reduce dense matrix dimensions"""
    #     matrix_size_gb = features.nbytes / (1024**3)
    #     logger.info(f"Processing dense matrix ({matrix_size_gb:.2f} GB)")
        
    #     # Convert large dense matrices to sparse if they have many zeros
    #     if matrix_size_gb > 0.5:  # More than 500MB
    #         sparsity = np.count_nonzero(features) / features.size
    #         logger.info(f"Matrix sparsity: {sparsity:.3f}")
            
    #         if sparsity < 0.1:  # Less than 10% non-zero
    #             logger.info("Converting dense to sparse due to high sparsity")
    #             sparse_features = scipy.sparse.csr_matrix(features)
    #             del features  # Free memory immediately
    #             gc.collect()
    #             return self._reduce_sparse_safely(sparse_features, n_components)
        
    #     # Use randomized SVD directly for better memory efficiency
    #     if matrix_size_gb > 1.0:  # More than 1GB
    #         logger.info("Using direct randomized SVD for large dense matrix")
    #         try:
    #             U, s, Vt = randomized_svd(
    #                 features, 
    #                 n_components=n_components,
    #                 n_iter=5,
    #                 random_state=42
    #             )
    #             reduced_features = U * s
                
    #             # Clean up
    #             del U, s, Vt, features
    #             gc.collect()
                
    #             logger.info(f"Direct SVD reduction complete. New shape: {reduced_features.shape}")
    #             return reduced_features
                
    #         except Exception as e:
    #             logger.error(f"Direct SVD failed: {e}")
    #             # Fall back to batch processing
    #             return self._batch_reduce_dense(features, n_components)
        
    #     # Standard PCA for smaller dense matrices
    #     if self.dimension_reducer is None or not isinstance(self.dimension_reducer, PCA):
    #         self.dimension_reducer = PCA(
    #             n_components=n_components,
    #             svd_solver='randomized',
    #             random_state=42
    #         )
            
    #     reduced_features = self.dimension_reducer.fit_transform(features)
    #     logger.info(f"PCA reduction complete. New shape: {reduced_features.shape}")
    #     return reduced_features

    # def _batch_reduce_sparse(self, features: scipy.sparse.spmatrix, n_components: int, batch_size: int = 2000) -> np.ndarray:
    #     """Process sparse matrix in batches to avoid memory issues"""
    #     logger.info(f"Batch processing sparse matrix with batch_size={batch_size}")
        
    #     n_samples = features.shape[0]
    #     if n_samples <= batch_size:
    #         # Single batch processing
    #         svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
    #         return svd.fit_transform(features)
        
    #     # Multiple batch processing - fit on first batch, transform rest
    #     batches = []
    #     svd = None
        
    #     for i in range(0, n_samples, batch_size):
    #         end_idx = min(i + batch_size, n_samples)
    #         batch = features[i:end_idx]
            
    #         if svd is None:
    #             # Fit on first batch
    #             svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
    #             reduced_batch = svd.fit_transform(batch)
    #         else:
    #             # Transform subsequent batches
    #             reduced_batch = svd.transform(batch)
                
    #         batches.append(reduced_batch)
            
    #         # Clean up batch
    #         del batch
    #         gc.collect()
            
    #         logger.debug(f"Processed batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
        
    #     # Combine all batches
    #     result = np.vstack(batches)
        
    #     # Store the fitted transformer
    #     self.dimension_reducer = svd
        
    #     # Clean up
    #     del batches
    #     gc.collect()
        
    #     return result

    # def _batch_reduce_dense(self, features: np.ndarray, n_components: int, batch_size: int = 1000) -> np.ndarray:
    #     """Process dense matrix in batches"""
    #     logger.info(f"Batch processing dense matrix with batch_size={batch_size}")
        
    #     n_samples = features.shape[0]
    #     if n_samples <= batch_size:
    #         # Use randomized SVD directly
    #         U, s, Vt = randomized_svd(features, n_components=n_components, random_state=42)
    #         return U * s
        
    #     # For very large matrices, we need to fit a global model first
    #     # Sample a subset for fitting
    #     sample_size = min(2000, n_samples)
    #     sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    #     sample_data = features[sample_indices]
        
    #     # Fit PCA on sample
    #     pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    #     pca.fit(sample_data)
        
    #     # Transform in batches
    #     batches = []
    #     for i in range(0, n_samples, batch_size):
    #         end_idx = min(i + batch_size, n_samples)
    #         batch = features[i:end_idx]
    #         reduced_batch = pca.transform(batch)
    #         batches.append(reduced_batch)
            
    #         del batch
    #         gc.collect()
        
    #     # Store the fitted transformer
    #     self.dimension_reducer = pca
        
    #     result = np.vstack(batches)
    #     del batches
    #     gc.collect()
        
    #     return result
