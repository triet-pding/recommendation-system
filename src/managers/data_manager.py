from recommendation_config import RecommendationConfig
import structlog
import mysql.connector
import redis.connection
from mysql.connector import Error
import time
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any
import redis
import re
import gc
import time
import psutil 
from datetime import timedelta
import os
from pathlib import Path
import json

logger = structlog.get_logger()

class DataSaveLoadError(Exception):
    """Custom exception for data save/load operations."""
    pass

class DataManager:
    """Handles data gathering and preprocessing"""
    def __init__(self, config: RecommendationConfig) -> None:
        self.config = config
        self.mysql_host = config.database_config.mysql_host
        self.mysql_name = config.database_config.mysql_name
        self.mysql_user = config.database_config.mysql_user
        self.mysql_password = config.database_config.mysql_password
        self.redis_host = config.database_config.redis_host
        self.redis_port = config.database_config.redis_port
        self.redis_password = config.database_config.redis_password

    def _extract_table_name(self, query: str) -> str:
        """
        Extract table name from a simple SELECT query.
        Assumes format: SELECT ... FROM table_name ...
        """
        match = re.search(r'from\s+([`"]?)(\w+)\1', query, re.IGNORECASE)
        return match.group(2) if match else f"table_{hash(query)}"
    
    def connect_to_redis(self) -> redis.Redis | None:
        """
        Create and return a connection to a Redis Cloud database.
        """
        try:
            redis_conn = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )

            # Test the connection
            if redis_conn.ping():
                logger.info("Connected to Redis successfully")
                return redis_conn
            else:
                logger.error("Redis ping failed.")
                return None

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            return None

    def connect_to_mysql(self) -> mysql.connector:
        """Creates and returns a connection to the MySQL database."""
        try:
            connection = mysql.connector.connect(
                host=self.mysql_host,          # Your MySQL server address (localhost for local)
                database=self.mysql_name,  # Your database name
                user=self.mysql_user,      # Your MySQL username
                password=self.mysql_password   # Your MySQL password
                # Uncomment below if needed:
                # port=3306,               # MySQL default port is 3306
                # auth_plugin='mysql_native_password'  # If using newer MySQL versions
            )
            
            if connection.is_connected():
                logger.info("Connected to MySQL database")
                return connection
                
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return None
    
    def close_connection(self, connection: mysql.connector):
        """
        Close the database connection.
        """
        if connection and connection.is_connected():
            connection.close()
            logger.info("MySQL connection closed")


    def execute_multiple_queries_with_timing(self, connection: mysql.connector.MySQLConnection, queries: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Execute multiple SQL queries sequentially and return a dictionary of pandas DataFrames keyed by table name.
        """
        results = {}
        total_start_time = time.perf_counter()

        try:
            cursor = connection.cursor(dictionary=True)
            for idx, query in enumerate(tqdm(queries, desc="Executing queries")):
                table_name = self._extract_table_name(query)
                query_start_time = time.perf_counter()

                try:
                    cursor.execute(query)
                    records = cursor.fetchall()
                    df = pd.DataFrame(records)
                    query_end_time = time.perf_counter()

                    if df.empty:
                        logger.warning(f"[{table_name}] returned no records (executed in {query_end_time - query_start_time:.4f} seconds).")
                    else:
                        logger.info(f"[{table_name}] executed in {query_end_time - query_start_time:.4f} seconds.")
                    
                    results[table_name] = df

                except Error as e:
                    logger.error(f"Error executing query {idx + 1} [{table_name}]: {e}")
                    results[table_name] = None

            cursor.close()

        except Error as e:
            logger.error(f"Error setting up cursor: {e}")

        total_end_time = time.perf_counter()
        logger.info(f"Total execution time: {total_end_time - total_start_time:.4f} seconds.")
        return results


    
    
    def built_master_set(self, vp_df: pd.DataFrame, v_df: pd.DataFrame,
                    vr_df: pd.DataFrame, uf_df: pd.DataFrame,
                    chunk_size: int = 10000) -> pd.DataFrame:
        """
        Build a master dataset from multiple video-related DataFrames in chunks,
        joining on relevant fields and logging performance and memory usage.
        """
        try:
            start_time = time.time()
            logger.info("Starting master set build process...")

            # Clean and prepare
            v_df = v_df[v_df.get('is_deleted', 0) == 0]
            uf_df = uf_df[uf_df.get('is_deleted', 0) == 0]
            vr_df['last_updated_date'] = pd.to_datetime(vr_df['updated_seconds'], unit='s', errors='coerce')

            v_df = v_df.rename(columns={'duration': 'video_duration'})
            vp_df = vp_df.rename(columns={
                'last_update_date': 'last_purchased_date',
                'duration': 'purchase_tier'
            })

            total_rows = len(vp_df)
            logger.info(f"Total rows to process: {total_rows}")
            logger.info(f"Memory usage at start: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")

            processed_chunks = []
            rows_processed = 0

            for chunk_start in range(0, total_rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_rows)
                try:
                    chunk_start_time = time.time()
                    vp_chunk = vp_df.iloc[chunk_start:chunk_end].copy()

                    temp_df = pd.merge(
                        vp_chunk,
                        v_df[['video_id', 'video_duration', 'rating_score', 'title', 'description']],
                        on='video_id',
                        how='inner'
                    )

                    if temp_df.empty:
                        logger.info(f"Chunk {chunk_start // chunk_size + 1}: No rows after first join. Skipping.")
                        continue

                    temp_df = pd.merge(
                        temp_df,
                        vr_df[['rating', 'video_id', 'user_id', 'last_updated_date']],
                        on=['user_id', 'video_id'],
                        how='inner'
                    )

                    if temp_df.empty:
                        logger.info(f"Chunk {chunk_start // chunk_size + 1}: No rows after second join. Skipping.")
                        continue

                    temp_df = pd.merge(
                        temp_df,
                        uf_df[['following', 'pd_category', 'pd_language']],
                        left_on='video_owner_user_id',
                        right_on='following',
                        how='inner'
                    )

                    if not temp_df.empty:
                        processed_chunks.append(temp_df.drop_duplicates())

                    rows_processed += len(vp_chunk)
                    chunk_time = time.time() - chunk_start_time
                    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

                    logger.info(f"Chunk {chunk_start // chunk_size + 1}: {chunk_time:.2f}s, memory: {mem_usage:.2f} MB, rows processed: {rows_processed}/{total_rows}")

                    del temp_df, vp_chunk
                    gc.collect()

                except Exception as chunk_err:
                    logger.exception(f"Exception in chunk {chunk_start // chunk_size + 1}: {chunk_err}")

            if not processed_chunks:
                logger.warning("No valid rows after processing all chunks.")
                return pd.DataFrame()

            final_df = pd.concat(processed_chunks, ignore_index=True).drop_duplicates()

            if 'last_purchased_date' in final_df.columns:
                final_df.sort_values(by='last_purchased_date', ascending=False, inplace=True) # sort the data by time

            elapsed_minutes = (time.time() - start_time) / 60
            logger.info(f"Master dataset built successfully. Shape: {final_df.shape}, Time: {elapsed_minutes:.2f} minutes")
            return final_df

        except Exception as e:
            logger.exception(f"Unexpected error in master set construction: {e}")
            return pd.DataFrame()
        
    def feature_selection_and_clean_up(self, master_df: pd.DataFrame) -> pd.DataFrame:

        # List of compulsory columns
        required_columns = ['title', 'description', 'trees_consumed', 
                            'video_duration', 'purchase_tier', 'pd_category']
            
        missing_columns = [col for col in required_columns if col not in master_df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle mssing data
        logger.info("Handling missing data...")

        master_df['title'] = master_df['title'].fillna("")
        master_df['description'] = master_df['description'].fillna("")
        master_df['trees_consumed'] = master_df['trees_consumed'].fillna(master_df['trees_consumed'].median())
        master_df['video_duration'] = master_df['video_duration'].fillna(master_df['video_duration'].median())
        master_df['purchase_tier'] = master_df['purchase_tier'].fillna(master_df['purchase_tier'].value_counts().index[0])
        master_df['pd_category'] = master_df['pd_category'].fillna(master_df['pd_category'].value_counts().index[0])

        # Dropping unwanted columns
        master_df.drop(columns=['drm_fee', 'discount_percentage_applied', 'package_purchase_id', 
                            'is_replacement_of_deleted_video', 'following','is_refunded', 
                            'expiry_date', 'id', 'video_owner_user_id'], inplace=True)
        master_df.rename(columns={'rating_score': 'wilson_score'}, inplace=True)
        master_df.drop_duplicates(inplace=True)

        return master_df

    def build_master_set_sql_optimized(self, connection: mysql.connector, chunk_size: int = 50000) -> pd.DataFrame:
        """
        Execute the optimized SQL query directly in the database with chunking.
        This is the most efficient approach as it leverages database optimizations.
        """
        try:
            start_time = time.time()
            logger.info("Starting SQL-optimized master set build...")
            
            # Your original SQL query with minor optimizations
            base_query = """
            SELECT DISTINCT
                vp.id,
                vp.last_update_date as last_purchased_date,
                FROM_UNIXTIME(vr.updated_seconds) as last_updated_date,
                vp.trees_consumed,
                vp.user_id,
                vp.video_id,
                vp.video_owner_user_id,
                vp.duration as purchase_tier,
                v.duration as video_duration,
                v.rating_score,
                v.title,
                v.description,
                uf.pd_category,
                uf.pd_language,
                vr.rating
            FROM pding_prod_db.video_purchase vp
            INNER JOIN pding_prod_db.videos v ON vp.video_id = v.video_id
            INNER JOIN pding_prod_db.video_rating vr ON vp.user_id = vr.user_id AND vp.video_id = vr.video_id
            INNER JOIN pding_prod_db.user_followings uf ON vp.video_owner_user_id = uf.following
            WHERE v.is_deleted = 0 
                AND vr.rating IS NOT NULL 
                AND uf.is_deleted = 0
            ORDER BY last_purchased_date DESC
            """
            
            # Get total count first for progress tracking
            count_query = f"""
            SELECT COUNT(*) as total_count FROM (
                {base_query.replace('ORDER BY last_purchased_date DESC', '')}
            ) as count_subquery
            """
            
            total_count = pd.read_sql(count_query, connection).iloc[0]['total_count']
            logger.info(f"Total rows to fetch: {total_count}")
            
            if total_count <= chunk_size:
                # Small dataset, fetch all at once
                df = pd.read_sql(base_query, connection)
            else:
                # Large dataset, use chunking with LIMIT/OFFSET
                chunks = []
                for offset in range(0, total_count, chunk_size):
                    chunk_query = f"{base_query} LIMIT {chunk_size} OFFSET {offset}"
                    chunk_df = pd.read_sql(chunk_query, connection)
                    chunks.append(chunk_df)
                    
                    logger.info(f"Fetched chunk: {offset//chunk_size + 1}, "
                                   f"rows: {len(chunk_df)}, "
                                   f"progress: {min(offset + chunk_size, total_count)}/{total_count}")
                
                df = pd.concat(chunks, ignore_index=True)
            
            connection.close()
            
            elapsed_time = time.time() - start_time
            logger.info(f"SQL-optimized build completed. Shape: {df.shape}, Time: {elapsed_time:.2f}s")
            return df
            
        except Exception as e:
            logger.exception(f"Error in SQL-optimized build: {e}")
            return pd.DataFrame()


    def load_data(self) -> Tuple[pd.DataFrame]:
        connection = self.connect_to_mysql()
        queries = [
            "select * from videos",
            "select * from video_rating",
            "select * from video_purchase",
            "select * from user_followings"
        ]
        dfs_dict = self.execute_multiple_queries_with_timing(connection, queries)

        # Retrieve component datafarmes
        videos_df = dfs_dict.get("videos")
        video_rating_df = dfs_dict.get("video_rating")
        video_purchase_df = dfs_dict.get("video_purchase")
        user_followings_df = dfs_dict.get("user_followings")

        self.close_connection(connection)

        master_df = self.built_master_set(vp_df=video_purchase_df,
                                                vr_df=video_rating_df,
                                                v_df=videos_df,
                                                uf_df=user_followings_df)

        #master_df = self.build_master_set_sql_optimized(connection=connection)

        master_df = self.feature_selection_and_clean_up(master_df)

        return master_df
    
    def temporal_split_recommendation_data(self, master_df: pd.DataFrame, split_strategy: str ='per_user', **kwargs) -> Tuple[pd.DataFrame]:
        """
        Perform temporal split on recommendation data using last_purchased_date
        
        Parameters:
        -----------
        master_df : Preprocessed DataFrame
            DataFrame with 'last_purchased_date' column (already sorted descending)a
        split_strategy : str
            'percentage' - split by percentage of time range
            'fixed_date' - split at specific date
            'days_back' - use last N days as test set
            'per_user' - split per user temporally
        **kwargs : additional parameters for each strategy
        
        Returns:
        --------
        train_df, test_df : tuple of DataFrames
        """
        
        # Ensure datetime format
        if master_df['last_purchased_date'].dtype != 'datetime64[ns]':
            master_df['last_purchased_date'] = pd.to_datetime(master_df['last_purchased_date'])
        
        
        logger.info(f"=== TEMPORAL SPLIT ANALYSIS ===")
        logger.info(f"Total records: {len(master_df)}")
        logger.info(f"Date range: {master_df['last_purchased_date'].min()} to {master_df['last_purchased_date'].max()}")
        logger.info(f"Time span: {(master_df['last_purchased_date'].max() - master_df['last_purchased_date'].min()).days} days")
        
        if split_strategy == 'percentage':
            return self._percentage_split(master_df, **kwargs)
        elif split_strategy == 'fixed_date':
            return self._fixed_date_split(master_df, **kwargs)
        elif split_strategy == 'days_back':
            return self._days_back_split(master_df, **kwargs)
        elif split_strategy == 'per_user':
            return self._per_user_temporal_split(master_df, **kwargs)
        else:
            raise ValueError("Invalid split_strategy. Choose: 'percentage', 'fixed_date', 'days_back', 'per_user'")

    def _percentage_split(self, df: pd.DataFrame, train_percentage: float =0.8) -> Tuple[pd.DataFrame]:
        """Split by percentage of time range"""
        min_date = df['last_purchased_date'].min()
        max_date = df['last_purchased_date'].max()
        time_range = max_date - min_date
        
        # Calculate cutoff date
        cutoff_date = min_date + (time_range * train_percentage)
        
        logger.info(f"PERCENTAGE SPLIT ({train_percentage*100}%/{(1-train_percentage)*100}%):")
        logger.info(f"Cutoff date: {cutoff_date}")
        
        # Split data
        train_df = df[df['last_purchased_date'] <= cutoff_date].copy()
        test_df = df[df['last_purchased_date'] > cutoff_date].copy()
        
        self._print_split_stats(train_df, test_df, cutoff_date)
        return train_df, test_df

    def _fixed_date_split(self, df: pd.DataFrame, cutoff_date: str) -> Tuple[pd.DataFrame]:
        """Split at specific date"""
        if isinstance(cutoff_date, str):
            cutoff_date = pd.to_datetime(cutoff_date)
        
        logger.info(f"FIXED DATE SPLIT:")
        logger.info(f"Cutoff date: {cutoff_date}")
        
        train_df = df[df['last_purchased_date'] <= cutoff_date].copy()
        test_df = df[df['last_purchased_date'] > cutoff_date].copy()
        
        self._print_split_stats(train_df, test_df, cutoff_date)
        return train_df, test_df

    def _days_back_split(self, df: pd.DataFrame, test_days: int=30) -> pd.DataFrame:
        """Use last N days as test set"""
        max_date = df['last_purchased_date'].max()
        cutoff_date = max_date - timedelta(days=test_days)
        
        logger.info(f"DAYS BACK SPLIT (last {test_days} days as test):")
        logger.info(f"Cutoff date: {cutoff_date}")
        
        train_df = df[df['last_purchased_date'] <= cutoff_date].copy()
        test_df = df[df['last_purchased_date'] > cutoff_date].copy()
        
        self._print_split_stats(train_df, test_df, cutoff_date)
        return train_df, test_df

    def _per_user_temporal_split(self, df: pd.DataFrame, test_percentage: float = 0.2, min_interactions: int = 5) -> Tuple[pd.DataFrame]:
        """Split per user: use each user's most recent interactions as test"""
        logger.info(f"PER-USER TEMPORAL SPLIT ({test_percentage*100}% recent per user):")
        logger.info(f"Minimum interactions per user: {min_interactions}")
        
        train_records = []
        test_records = []
        users_processed = 0
        users_skipped = 0
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].sort_values('last_purchased_date', ascending=False)
            
            if len(user_data) < min_interactions:
                # If user has too few interactions, put all in training
                train_records.append(user_data)
                users_skipped += 1
                continue
            
            # Calculate split point for this user
            n_test = max(1, int(len(user_data) * test_percentage))
            
            # Most recent interactions go to test
            user_test = user_data.iloc[:n_test]
            user_train = user_data.iloc[n_test:]
            
            train_records.append(user_train)
            test_records.append(user_test)
            users_processed += 1
        
        train_df = pd.concat(train_records, ignore_index=True) if train_records else pd.DataFrame()
        test_df = pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame()
        
        logger.info(f"Users processed: {users_processed}")
        logger.info(f"Users skipped (< {min_interactions} interactions): {users_skipped}")
        self._print_split_stats(train_df, test_df, None)
        
        return train_df, test_df

    def _print_split_stats(self, train_df: pd.DataFrame, test_df: pd.DataFrame, cutoff_date: Optional[str]):
        """
        Provide statistics data about the split strategy
        
        Parameters:
        -----------
        train_df : Training set DataFrame
        test_df : Testing set DataFrame
        cutoff_date: 
            String pattern sample: 2025-01-31 17:18:21.630378600
        
        Returns:
        --------
        None
        """
        logger.info(f"\nSPLIT RESULTS:")
        logger.info(f"Training set: {len(train_df)} records ({len(train_df)/(len(train_df)+len(test_df))*100:.1f}%)")
        logger.info(f"Test set: {len(test_df)} records ({len(test_df)/(len(train_df)+len(test_df))*100:.1f}%)")
        
        # Show cutoff date if provided
        if cutoff_date is not None:
            logger.info(f"Split cutoff date: {cutoff_date}")
        
        if len(train_df) > 0:
            logger.info(f"Training date range: {train_df['last_purchased_date'].min()} to {train_df['last_purchased_date'].max()}")
        if len(test_df) > 0:
            logger.info(f"Test date range: {test_df['last_purchased_date'].min()} to {test_df['last_purchased_date'].max()}")
        
        # Validate temporal ordering
        if cutoff_date is not None and len(train_df) > 0 and len(test_df) > 0:
            if train_df['last_purchased_date'].max() > cutoff_date:
                logger.warning("WARNING: Some training data is after cutoff date!")
            if test_df['last_purchased_date'].min() <= cutoff_date:
                logger.warning("WARNING: Some test data is before cutoff date!")
        
        # User overlap analysis
        train_users = set(train_df['user_id'].unique()) if len(train_df) > 0 else set()
        test_users = set(test_df['user_id'].unique()) if len(test_df) > 0 else set()
        
        logger.info(f"\nUSER ANALYSIS:")
        logger.info(f"Training users: {len(train_users)}")
        logger.info(f"Test users: {len(test_users)}")
        logger.info(f"Overlapping users: {len(train_users & test_users)}")
        logger.info(f"Cold start users (test only): {len(test_users - train_users)}")
        
        # Item overlap analysis
        if 'video_id' in train_df.columns:
            train_items = set(train_df['video_id'].unique()) if len(train_df) > 0 else set()
            test_items = set(test_df['video_id'].unique()) if len(test_df) > 0 else set()
            
            logger.info(f"\nITEM ANALYSIS:")
            logger.info(f"Training items: {len(train_items)}")
            logger.info(f"Test items: {len(test_items)}")
            logger.info(f"Overlapping items: {len(train_items & test_items)}")
            logger.info(f"Cold start items (test only): {len(test_items - train_items)}")

    
    def save_split_data(self, model_type: str, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                    save_dir: Union[str, Path], file_format: str = 'csv', 
                    compression: Optional[str] = None) -> Dict[str, Any]:
        """
        Save train and test dataframes to specified directory with comprehensive error handling
        and metadata generation.
        
        Parameters:
        -----------
        model_type: type of model, either cf (Collaborative Filtering) or cbf (Content-Based Filtering)
        train_df : pandas.DataFrame
            Training dataset
        test_df : pandas.DataFrame
            Test dataset
        save_dir : str or Path
            Directory path to save the files
        file_format : str, default='csv'
            File format: 'csv', 'parquet', 'pickle', 'json'
        compression : str, optional
            Compression method: 'gzip', 'bz2', 'xz' (for csv/json), 'snappy', 'gzip' (for parquet)
        
        Returns:
        --------
        dict : Paths of saved files and metadata with success status
        """
        
        # Validate inputs
        if train_df.empty or test_df.empty:
            raise DataSaveLoadError("Cannot save empty dataframes")
        
        save_dir = Path(save_dir)
        
        # Validate compression for file format
        valid_compressions = {
            'csv': ['gzip', 'bz2', 'xz', None],
            'parquet': ['snappy', 'gzip', 'brotli', None],
            'pickle': ['gzip', 'bz2', 'xz', None],
            'json': ['gzip', 'bz2', 'xz', None]
        }
        
        if file_format not in valid_compressions:
            raise DataSaveLoadError(f"Unsupported format: {file_format}. Choose from: {list(valid_compressions.keys())}")
        
        if compression and compression not in valid_compressions[file_format]:
            raise DataSaveLoadError(f"Unsupported compression '{compression}' for format '{file_format}'. "
                                f"Valid options: {valid_compressions[file_format]}")
        
        # Create directory if it doesn't exist
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise DataSaveLoadError(f"Failed to create directory {save_dir}: {e}")
        
        # Define file extensions based on compression
        def get_extension(fmt: str, comp: Optional[str]) -> str:
            base_ext = {
                'csv': '.csv',
                'parquet': '.parquet', 
                'pickle': '.pkl',
                'json': '.json'
            }[fmt]
            
            if comp == 'gzip':
                return base_ext + '.gz'
            elif comp == 'bz2':
                return base_ext + '.bz2'
            elif comp == 'xz':
                return base_ext + '.xz'
            else:
                return base_ext
        
        # Define save functions with proper error handling
        def get_save_function(fmt: str):
            save_functions = {
                'csv': lambda df, path, comp: df.to_csv(path, index=False, compression=comp),
                'parquet': lambda df, path, comp: df.to_parquet(path, compression=comp, index=False),
                'pickle': lambda df, path, comp: df.to_pickle(path, compression=comp),
                'json': lambda df, path, comp: df.to_json(path, orient='records', compression=comp)
            }
            return save_functions[fmt]
        
        # Get file extension and save function
        ext = get_extension(file_format, compression)
        save_func = get_save_function(file_format)
        
        # Define file paths
        train_filename = f"{model_type}_train_data_{ext}"
        test_filename = f"{model_type}_test_data_{ext}"
        metadata_filename = f"{model_type}_split_metadata.json"
        
        train_path = save_dir / train_filename
        test_path = save_dir / test_filename
        metadata_path = save_dir / metadata_filename
        
        logger.info("=== SAVING SPLIT DATA ===")
        logger.info("Save directory", directory=str(save_dir))
        logger.info("File format", format=file_format)
        if compression:
            logger.info("Compression", method=compression)
        
        try:
            # Save training data
            logger.info("Saving training data", filename=train_filename)
            save_func(train_df, train_path, compression)
            
            # Save test data
            logger.info("Saving test data", filename=test_filename)
            save_func(test_df, test_path, compression)
            
            # Calculate file sizes
            train_size_mb = train_path.stat().st_size / (1024 * 1024)
            test_size_mb = test_path.stat().st_size / (1024 * 1024)
            
            # Create comprehensive metadata
            metadata = {
                'file_format': file_format,
                'compression': compression,
                'file_extension': ext,
                'train_records': len(train_df),
                'test_records': len(test_df),
                'train_file_size_mb': round(train_size_mb, 2),
                'test_file_size_mb': round(test_size_mb, 2),
                'total_size_mb': round(train_size_mb + test_size_mb, 2),
                'train_filename': train_filename,
                'test_filename': test_filename,
                'train_columns': list(train_df.columns),
                'test_columns': list(test_df.columns),
                'column_dtypes': {col: str(dtype) for col, dtype in train_df.dtypes.items()},
            }
            
            # Add dataset-specific metadata
            dataset_metadata = {}
            for name, df in [('train', train_df), ('test', test_df)]:
                meta = {}
                if 'user_id' in df.columns:
                    meta['unique_users'] = int(df['user_id'].nunique())
                if 'video_id' in df.columns:
                    meta['unique_items'] = int(df['video_id'].nunique())
                if 'last_purchased_date' in df.columns and len(df) > 0:
                    meta['date_range'] = {
                        'min': str(df['last_purchased_date'].min()),
                        'max': str(df['last_purchased_date'].max())
                    }
                dataset_metadata[name] = meta
            
            metadata['dataset_stats'] = dataset_metadata
            
            # Add loading instructions
            loading_instructions = {
                'python_code': self.get_loading_code(train_filename, test_filename, file_format, compression),
                'required_libraries': ['pandas'] + (['pyarrow'] if file_format == 'parquet' else [])
            }
            metadata['loading_instructions'] = loading_instructions
            
            # Save metadata as JSON for easier parsing
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info("Saving metadata", filename=metadata_filename)
            logger.info("Files saved successfully!")
            logger.info("Training data", records=len(train_df), size_mb=f"{train_size_mb:.2f}")
            logger.info("Test data", records=len(test_df), size_mb=f"{test_size_mb:.2f}")
            logger.info("Total size", mb=f"{train_size_mb + test_size_mb:.2f}")
            
            return {
                'train_path': str(train_path),
                'test_path': str(test_path),
                'metadata_path': str(metadata_path),
                'metadata': metadata,
                'success': True
            }
            
        except Exception as e:
            logger.error("Error saving data", error=str(e))
            # Cleanup partial files
            for path in [train_path, test_path, metadata_path]:
                if path.exists():
                    try:
                        path.unlink()
                    except:
                        pass
            
            raise DataSaveLoadError(f"Failed to save data: {e}")


    def load_split_data(self, 
                        model_type: str, 
                        save_dir: Union[str, Path], 
                        file_format: Optional[str] = None, 
                        compression: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Load previously saved train/test data with automatic format and compression detection.
        
        Parameters:
        -----------
        model_type: str
        save_dir : str or Path
            Directory containing the saved files
        file_format : str, optional
            File format to load. If None, auto-detects from metadata
        compression : str, optional
            Compression method. If None, auto-detects from filename/metadata
        
        Returns:
        --------
        tuple : (train_df, test_df, metadata_dict)
        """
        
        save_dir = Path(save_dir)
        
        if not save_dir.exists():
            raise DataSaveLoadError(f"Directory not found: {save_dir}")
        
        try:
            # First, try to find and load metadata
            metadata = None
            metadata_file = None
            
            metadata_files = sorted(save_dir.glob(f"{model_type}_split_metadata_*.json"))
            
            if metadata_files:
                metadata_file = metadata_files[-1]  # Most recent
                logger.info("Loading metadata", file=metadata_file.name)
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Extract info from metadata
                if not file_format:
                    file_format = metadata['file_format']
                if not compression:
                    compression = metadata.get('compression')
            
            # Find data files
            train_pattern = f"{model_type}_train_data_*"
            test_pattern = f"{model_type}_test_data_*"
            
            train_files = sorted(save_dir.glob(train_pattern))
            test_files = sorted(save_dir.glob(test_pattern))
            
            if not train_files or not test_files:
                raise DataSaveLoadError("No matching train/test files found")
            
            # Use most recent files
            train_file = train_files[-1]
            test_file = test_files[-1]
            
            # Auto-detect format and compression from filename if not provided
            if not file_format or not compression:
                detected_format, detected_compression = self.detect_format_and_compression(train_file)
                file_format = file_format or detected_format
                compression = compression or detected_compression
            
            logger.info("Loading train data", file=train_file.name, format=file_format, compression=compression)
            logger.info("Loading test data", file=test_file.name, format=file_format, compression=compression)
            
            # Load data based on format with proper compression handling
            train_df = self.load_dataframe(train_file, file_format, compression)
            test_df = self.load_dataframe(test_file, file_format, compression)
            
            logger.info("Data loaded successfully!")
            logger.info("Training data", records=len(train_df))
            logger.info("Test data", records=len(test_df))
            
            return train_df, test_df, metadata or {}
            
        except Exception as e:
            logger.error("Error loading data", error=str(e))
            raise DataSaveLoadError(f"Failed to load data: {e}")


    def detect_format_and_compression(self, file_path: Path) -> Tuple[str, Optional[str]]:
        """
        Detect file format and compression from filename.
        
        Returns:
        --------
        tuple : (file_format, compression)
        """
        filename = file_path.name.lower()
        
        # Detect compression first
        compression = None
        if filename.endswith('.gz'):
            compression = 'gzip'
            filename = filename[:-3]  # Remove .gz
        elif filename.endswith('.bz2'):
            compression = 'bz2'
            filename = filename[:-4]  # Remove .bz2
        elif filename.endswith('.xz'):
            compression = 'xz'
            filename = filename[:-3]  # Remove .xz
        
        # Detect format
        if filename.endswith('.csv'):
            return 'csv', compression
        elif filename.endswith('.parquet'):
            return 'parquet', compression
        elif filename.endswith('.pkl'):
            return 'pickle', compression
        elif filename.endswith('.json'):
            return 'json', compression
        else:
            raise DataSaveLoadError(f"Cannot detect format from filename: {file_path.name}")


    def load_dataframe(self, file_path: Path, file_format: str, compression: Optional[str]) -> pd.DataFrame:
        """
        Load dataframe with proper format and compression handling.
        
        Parameters:
        -----------
        file_path : Path
            Path to the file
        file_format : str
            File format ('csv', 'parquet', 'pickle', 'json')
        compression : str, optional
            Compression method
        
        Returns:
        --------
        pandas.DataFrame : Loaded dataframe
        """
        try:
            if file_format == 'csv':
                return pd.read_csv(file_path, compression=compression)
            elif file_format == 'parquet':
                return pd.read_parquet(file_path)  # Parquet handles compression automatically
            elif file_format == 'pickle':
                return pd.read_pickle(file_path, compression=compression)
            elif file_format == 'json':
                return pd.read_json(file_path, compression=compression)
            else:
                raise DataSaveLoadError(f"Unsupported format: {file_format}")
        
        except Exception as e:
            raise DataSaveLoadError(f"Failed to load {file_format} file {file_path.name}: {e}")


    def get_loading_code(self, train_filename: str, test_filename: str, 
                        file_format: str, compression: Optional[str]) -> Dict[str, str]:
        """Generate loading code examples for different formats."""
        
        code_templates = {
            'csv': {
                'train': f"train_df = pd.read_csv('{train_filename}'" + 
                        (f", compression='{compression}'" if compression else "") + ")",
                'test': f"test_df = pd.read_csv('{test_filename}'" + 
                    (f", compression='{compression}'" if compression else "") + ")"
            },
            'parquet': {
                'train': f"train_df = pd.read_parquet('{train_filename}')",
                'test': f"test_df = pd.read_parquet('{test_filename}')"
            },
            'pickle': {
                'train': f"train_df = pd.read_pickle('{train_filename}'" + 
                        (f", compression='{compression}'" if compression else "") + ")",
                'test': f"test_df = pd.read_pickle('{test_filename}'" + 
                    (f", compression='{compression}'" if compression else "") + ")"
            },
            'json': {
                'train': f"train_df = pd.read_json('{train_filename}'" + 
                        (f", compression='{compression}'" if compression else "") + ")",
                'test': f"test_df = pd.read_json('{test_filename}'" + 
                    (f", compression='{compression}'" if compression else "") + ")"
            }
        }
        
        return code_templates.get(file_format, {})
    
    def extract_unique_video_content(self, interaction_data: pd.DataFrame, preserve_temporal: bool = False) -> pd.DataFrame:
        """
        Extract unique video content from interaction data.
        
        For content-based filtering, we only need one record per video with its content features.
        This method handles the aggregation strategy for cases where video metadata might differ
        across interactions.
        
        Args:
            interaction_data: DataFrame with user-video interactions containing video metadata
            preserve_temporal: If True, preserves temporal information for later temporal splitting
            
        Returns:
            DataFrame with unique videos and their consolidated content features
        """
        logger.info(f"Extracting unique video content from {len(interaction_data)} interactions...")
        
        # Define content columns (features that describe the video content)
        content_columns = ['video_id', 'title', 'description', 'trees_consumed', 
                            'video_duration', 'purchase_tier', 'pd_category']
        
        # Add temporal column if preserving temporal info
        if preserve_temporal and 'last_purchased_date' in interaction_data.columns:
            content_columns.append('last_purchased_date')
            # Also preserve user_id for potential per-user temporal splitting
            if 'user_id' in interaction_data.columns:
                content_columns.append('user_id')
        
        # Check which content columns are available
        available_content_cols = [col for col in content_columns if col in interaction_data.columns]
        missing_cols = [col for col in content_columns if col not in interaction_data.columns]
        
        if missing_cols:
            logger.warning(f"Missing content columns: {missing_cols}")
        
        if 'video_id' not in available_content_cols:
            raise ValueError("video_id column is required")
        
        logger.info(f"Using content columns: {available_content_cols}")
        
        # Extract only the content columns
        content_data = interaction_data[available_content_cols].copy()
        
        # Strategy for handling potential inconsistencies in video metadata across interactions
        unique_videos = []
        
        for video_id in content_data['video_id'].unique():
            video_interactions = content_data[content_data['video_id'] == video_id]
            
            if len(video_interactions) == 1:
                # Single interaction - use as is
                unique_videos.append(video_interactions.iloc[0])
            else:
                # Multiple interactions - need aggregation strategy
                consolidated_video = self._consolidate_video_metadata(video_interactions, preserve_temporal)
                unique_videos.append(consolidated_video)
        
        unique_videos_df = pd.DataFrame(unique_videos).reset_index(drop=True)
        
        logger.info(f"Extracted {len(unique_videos_df)} unique videos from {len(interaction_data)} interactions")
        
        # Log some statistics about consolidation
        duplicate_videos = len(interaction_data) - len(unique_videos_df)
        if duplicate_videos > 0:
            logger.info(f"Consolidated {duplicate_videos} duplicate video entries")
            
        if preserve_temporal and 'last_purchased_date' in unique_videos_df.columns:
            logger.info(f"Preserved temporal information for temporal splitting")
            
        return unique_videos_df

    def _consolidate_video_metadata(self, video_interactions: pd.DataFrame, preserve_temporal: bool = False) -> pd.Series:
        """
        Consolidate metadata for a video that appears in multiple interactions.
        
        Strategy:
        - For text fields (title, description): Use the most recent non-null value
        - For numerical fields: Use the most common value, or mean if all different
        - For categorical fields: Use the most frequent value
        - For temporal fields: Use most recent date when preserve_temporal=True
        
        Args:
            video_interactions: All interactions for a single video_id
            preserve_temporal: If True, preserves the most recent temporal information
            
        Returns:
            Consolidated video metadata as a Series
        """
        consolidated = {}
        video_id = video_interactions['video_id'].iloc[0]
        consolidated['video_id'] = video_id
        
        for column in video_interactions.columns:
            if column == 'video_id':
                continue
                
            # Get non-null values
            non_null_values = video_interactions[column].dropna()
            
            if len(non_null_values) == 0:
                consolidated[column] = None
                continue
            
            # Consolidation strategy based on column type
            if column in ['title', 'description']:
                # For text: use the most recent non-null value (last occurrence)
                consolidated[column] = non_null_values.iloc[-1]
                
            elif column == 'last_purchased_date' and preserve_temporal:
                # For temporal data: use the most recent date
                consolidated[column] = non_null_values.max()
                
            elif column == 'user_id' and preserve_temporal:
                # For user_id when preserving temporal: use the user with most recent interaction
                if 'last_purchased_date' in video_interactions.columns:
                    # Find the user_id associated with the most recent purchase
                    most_recent_idx = video_interactions['last_purchased_date'].idxmax()
                    consolidated[column] = video_interactions.loc[most_recent_idx, 'user_id']
                else:
                    # Fallback to most frequent user
                    consolidated[column] = non_null_values.mode().iloc[0] if len(non_null_values.mode()) > 0 else non_null_values.iloc[0]
                
            elif column in ['trees_consumed', 'video_duration']:
                # For numerical: use most common value, or mean if all different
                value_counts = non_null_values.value_counts()
                if len(value_counts) == 1 or value_counts.iloc[0] > 1:
                    # All same or clear majority
                    consolidated[column] = value_counts.index[0]
                else:
                    # All different - use mean
                    consolidated[column] = non_null_values.mean()
                    
            elif column in ['purchase_tier', 'pd_category']:
                # For categorical: use most frequent value
                consolidated[column] = non_null_values.mode().iloc[0] if len(non_null_values.mode()) > 0 else non_null_values.iloc[0]
                
            else:
                # Default: use most frequent value
                consolidated[column] = non_null_values.mode().iloc[0] if len(non_null_values.mode()) > 0 else non_null_values.iloc[0]
        
        return pd.Series(consolidated)
    

    def extract_and_split_video_content(self, interaction_data: pd.DataFrame, split_strategy: str = 'per_user', **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract unique video content and perform temporal split in one step.
        
        This method combines the video content extraction with temporal splitting,
        ensuring that temporal information is preserved during the extraction process.
        
        Args:
            interaction_data: DataFrame with user-video interactions containing video metadata
            split_strategy: Same as temporal_split_recommendation_data
            **kwargs: Additional parameters for the split strategy
            
        Returns:
            Tuple of (train_df, test_df) with unique video content
        """
        logger.info("=== EXTRACTING UNIQUE VIDEO CONTENT WITH TEMPORAL SPLIT ===")
        
        # First extract unique video content while preserving temporal information
        unique_videos_df = self.extract_unique_video_content(interaction_data, preserve_temporal=True)
        
        # Check if we have the required temporal column
        if 'last_purchased_date' not in unique_videos_df.columns:
            raise ValueError("Cannot perform temporal split: 'last_purchased_date' column not found in extracted content")
        
        # Now perform temporal split on the unique video content
        train_df, test_df = self.temporal_split_recommendation_data(unique_videos_df, split_strategy, **kwargs)
        
        logger.info("=== CONTENT EXTRACTION AND TEMPORAL SPLIT COMPLETED ===")
        return train_df, test_df

    def prepare_content_based_data(self, interaction_data: pd.DataFrame, 
                                split_strategy: str = 'per_user', 
                                remove_temporal_after_split: bool = True,
                                **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data specifically for content-based filtering with temporal split.
        
        This method:
        1. Extracts unique video content with temporal information
        2. Performs temporal split
        3. Optionally removes temporal columns for pure content-based modeling
        
        Args:
            interaction_data: DataFrame with user-video interactions
            split_strategy: Temporal split strategy
            remove_temporal_after_split: If True, removes temporal columns after splitting
            **kwargs: Additional parameters for the split strategy
            
        Returns:
            Tuple of (train_df, test_df) ready for content-based filtering
        """
        logger.info("=== PREPARING CONTENT-BASED DATA WITH TEMPORAL SPLIT ===")
        
        # Extract and split with temporal information
        train_df, test_df = self.extract_and_split_video_content(interaction_data, split_strategy, **kwargs)
        
        if remove_temporal_after_split:
            # Remove temporal columns for pure content-based modeling
            temporal_cols = ['last_purchased_date', 'user_id']
            
            for col in temporal_cols:
                if col in train_df.columns:
                    logger.info(f"Removing temporal column '{col}' from training set")
                    train_df = train_df.drop(columns=[col])
                if col in test_df.columns:
                    logger.info(f"Removing temporal column '{col}' from test set")
                    test_df = test_df.drop(columns=[col])
        
        logger.info("=== CONTENT-BASED DATA PREPARATION COMPLETED ===")
        logger.info(f"Training set: {len(train_df)} unique videos")
        logger.info(f"Test set: {len(test_df)} unique videos")
        
        return train_df, test_df