import mysql.connector
import structlog
from mysql.connector import Error
import time
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import os
from typing import List
load_dotenv()
logger = structlog.get_logger()

DB_HOST=os.getenv('DB_HOST')            # Your database host
DB_NAME=os.getenv('DB_NAME')            # Your database name
DB_USER=os.getenv('DB_USER')            # Your MySQL username
DB_PASSWORD=os.getenv('DB_PASSWORD')   # Your MySQL password

def connect_to_mysql(
) -> mysql.connector:
    """
    Creates and returns a connection to the MySQL database.
    """
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,          # Your MySQL server address (localhost for local)
            database=DB_NAME,  # Your database name
            user=DB_USER,      # Your MySQL username
            password=DB_PASSWORD   # Your MySQL password
            # Uncomment below if needed:
            # port=3306,               # MySQL default port is 3306
            # auth_plugin='mysql_native_password'  # If using newer MySQL versions
        )
        
        if connection.is_connected():
            logger.info("Connected to MySQL database")
            return connection
            
    except Error as e:
        logger.info(f"Error connecting to MySQL: {e}")
        return None
    
def close_connection(connection: mysql.connector):
    """
    Close the database connection.
    """
    if connection and connection.is_connected():
        connection.close()
        logger.info("MySQL connection closed")


def execute_multiple_queries_with_timing(connection: mysql.connector, queries: List[str]) -> List[pd.DataFrame]:
    """
    Execute multiple SQL queries sequentially, return results as a list of pandas DataFrames,
    and track execution time for each query and the total process.
    """
    
    dataframes = []
    total_start_time = time.perf_counter()
    try:
        cursor = connection.cursor(dictionary=True)
        for idx, query in enumerate(tqdm(queries, desc="Executing queries")):
            
            query_start_time = time.perf_counter()
            try:
                cursor.execute(query)
                records = cursor.fetchall()
                df = pd.DataFrame(records)
                dataframes.append(df)
                query_end_time = time.perf_counter()
                logger.info(f"Query {idx + 1} executed in {query_end_time - query_start_time:.4f} seconds.")
            except Error as e:
                logger.info(f"Error executing query {idx + 1}: {e}")
                dataframes.append(None)
        cursor.close()
    except Error as e:
        logger.info(f"Error setting up cursor: {e}")
    total_end_time = time.perf_counter()
    logger.info(f"Total execution time: {total_end_time - total_start_time:.4f} seconds.")
    return dataframes
    

if __name__ == "__main__":
    connection = connect_to_mysql()
    queries = [
        "select * from videos"
    ]
    dfs = execute_multiple_queries_with_timing(connection, queries)
    videos_df = dfs[0]
    logger.info(f"Shape of videos_df: {videos_df.shape}")