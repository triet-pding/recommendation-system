import os
import pickle
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from konlpy.tag import Okt  # Korean language processor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import time
import faiss

logger = structlog.get_logger()
data_dir = Path(__file__).resolve().parent.parent / "data"


class ContentBasedRecommender:
    """
    Content-based filtering recommendation system for items with Korean metadata.
    Specifically handles Korean text in 'title' and 'description' attributes.
    """
    
    def __init__(self):
        #  Initialize Korean text processor
        self.okt = Okt() 
        self.korean_stopwords = self._load_korean_stopwords()
        
        # Vector database
        self.index = None
        self.id_mapping = {}
        
        # Initialize transformers
        self.text_vectorizer = None
        self.numerical_scaler = MinMaxScaler()
        self.categorical_encoder = OneHotEncoder(handle_unknown='ignore')

        logger.info("Content-based Recommender initialized")
        
    def _load_korean_stopwords(self):
        """Load Korean stopwords or use a default set if file not available"""
        logger.info("Load Korean stopwords list")
        try:
            with open(f'{data_dir}/korean_stopwords.txt', 'r', encoding='utf-8') as f:
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
        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")
        return text_features
    
    def extract_metadata_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and encode numerical and categorical metadata features"""
        logger.info("Extracting metadata features")
        start_time = time.perf_counter()
        # Handle numerical features
        numerical_features = df[['tree_consumed', 'video_duration']].values
        scaled_numerical = self.numerical_scaler.fit_transform(numerical_features)
        
        # Handle categorical features
        categorical_features = df[['purchase_tier', 'pd_category']].values
        encoded_categorical = self.categorical_encoder.fit_transform(categorical_features).toarray()
        
        # Combine all metadata features
        metadata_features = np.hstack((scaled_numerical, encoded_categorical))

        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")
        return metadata_features
    
    def combine_features(self, text_features: np.ndarray, metadata_features: np.ndarray, 
                         text_weight: float = 0.7) -> np.ndarray:
        """Combine text and metadata features with weighting"""
        logger.info(f"Combining features with text_weight={text_weight}")
        start_time = time.perf_counter()
        # Normalize feature matrices
        text_norm = np.sqrt((text_features.toarray() ** 2).sum(axis=1))
        text_normalized = text_features.toarray() / text_norm[:, np.newaxis]
        
        metadata_norm = np.sqrt((metadata_features ** 2).sum(axis=1))
        metadata_normalized = metadata_features / metadata_norm[:, np.newaxis]
        
        # Combine with weights
        combined_features = (text_weight * text_normalized + 
                             (1 - text_weight) * metadata_normalized)
        
        end_time = time.perf_counter()
        logger.info(f"Total execution time: {end_time - start_time:.4f} seconds.")
        return combined_features
    
    def build_faiss_index(self, feature_matrix: np.ndarray) -> None:
        """Build FAISS index for fast similarity search"""
        logger.info(f"Building FAISS index with {feature_matrix.shape[0]} videos")
        
        # Convert to float32 as required by FAISS
        features_float32 = feature_matrix.astype(np.float32)
        
        # Create and train index
        dimension = features_float32.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        
        # Add vectors to the index
        self.index.add(features_float32)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

    def fit(self, video_data: pd.DataFrame) -> None:
        """Fit the recommendation model on the provided video data"""
        logger.info(f"Fitting model on {len(video_data)} videos")
        
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

    def find_similar_videos(self, video_id: int, video_data: pd.DataFrame, top_n: int = 10) -> List[Tuple[int, float]]:
        """Find videos similar to the given video ID"""
        # # Cache key for this query
        # cache_key = f"sim_videos:{video_id}:{top_n}"
        
        # # Try to get from cache first
        # if self.use_cache:
        #     cached_result = self.cache.get(cache_key)
        #     if cached_result:
        #         CACHE_HIT_COUNTER.inc()
        #         logger.info(f"Cache hit for video_id={video_id}")
        #         return pickle.loads(cached_result)
        
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
            
            # Cache the result
            # if self.use_cache:
            #     self.cache.setex(cache_key, self.cache_ttl, pickle.dumps(similar_videos))
            
            return similar_videos
        
        except Exception as e:
            logger.error(f"Error finding similar videos: {str(e)}")
            return []
        
    def save_models(self) -> None:
        """Save trained models and preprocessors to disk"""
        logger.info(f"Saving models to {self.model_dir}")
        
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
        
        # Save ID mapping
        with open(os.path.join(self.model_dir, "id_mapping.pkl"), "wb") as f:
            pickle.dump(self.id_mapping, f)
            
        logger.info("Models saved successfully")
    
    def load_models(self) -> None:
        """Load trained models and preprocessors from disk"""
        logger.info(f"Loading models from {self.model_dir}")
        
        try:
            # Load text vectorizer
            with open(os.path.join(self.model_dir, "text_vectorizer.pkl"), "rb") as f:
                self.text_vectorizer = pickle.load(f)
            
            # Load numerical scaler
            with open(os.path.join(self.model_dir, "numerical_scaler.pkl"), "rb") as f:
                self.numerical_scaler = pickle.load(f)
            
            # Load categorical encoder
            with open(os.path.join(self.model_dir, "categorical_encoder.pkl"), "rb") as f:
                self.categorical_encoder = pickle.load(f)
            
            # Load FAISS index
            self.index = faiss.read_index(os.path.join(self.model_dir, "faiss_index.bin"))
            
            # Load ID mapping
            with open(os.path.join(self.model_dir, "id_mapping.pkl"), "rb") as f:
                self.id_mapping = pickle.load(f)
                
            logger.info("Models loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
        

if __name__ == "__main__":
    
    print(data_dir)