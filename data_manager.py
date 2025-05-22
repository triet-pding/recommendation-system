from recommendation_config import RecommendationConfig
import structlog
import mysql.connector
import redis.connection
from mysql.connector import Error
import time
import pandas as pd
from tqdm import tqdm
import os
from typing import List, Dict
import redis
import re
import gc
import time
import psutil 

logger = structlog.get_logger()

class DataManager:
    """Handles data gathering and preprocessing"""
    def __init__(self, config: RecommendationConfig) -> None:
        self.config = config
        self.mysql_host = config.get('mysql_host')
        self.mysql_name = config.get('mysql_name')
        self.mysql_user = config.get('mysql_user')
        self.mysql_password = config.get('mysql_password')
        self.redis_host = config.get('redis_host')
        self.redis_port = config.get('redis_port')
        self.redis_password = config.get('redis_password')

    def _extract_table_name(self, query: str) -> str:
        """
        Extract table name from a simple SELECT query.
        Assumes format: SELECT ... FROM table_name ...
        """
        match = re.search(r'from\s+([`"]?)(\w+)\1', query, re.IGNORECASE)
        return match.group(2) if match else f"table_{hash(query)}"

    def connect_to_mysql(self) -> mysql.connector:
        """Creates and returns a connection to the MySQL database."""
        try:
            connection = mysql.connector.connect(
                host=self.mysql_host,          # Your MySQL server address (localhost for local)
                database=self.mysql_name,  # Your database name
                user=self.mysql_name,      # Your MySQL username
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
                final_df.sort_values(by='last_purchased_date', ascending=False, inplace=True)

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
        logger.info(f"Raw master set: {master_df.info()}")

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

        logger.info(f"Preprocessed master set: {master_df.info()}")

        return master_df


    def load_data(self) -> pd.DataFrame:
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

        master_df = self.feature_selection_and_clean_up(master_df)
        return master_df