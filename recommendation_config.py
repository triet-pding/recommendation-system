from pathlib import Path
import structlog
import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

logger = structlog.get_logger()
load_dotenv()

MYSQL_HOST=str(os.getenv('MYSQL_DB_HOST'))         
MYSQL_NAME=str(os.getenv('MYSQL_DB_NAME'))          
MYSQL_USER=str(os.getenv('MYSQL_DB_USER'))           
MYSQL_PASSWORD=str(os.getenv('MYSQL_DB_PASSWORD'))

REDIS_HOST = os.getenv("REDIS_DB_HOST")      
REDIS_PORT = int(str(os.getenv("REDIS_DB_PORT")))  
REDIS_PASSWORD = os.getenv("REDIS_DB_PASSWORD")

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config" / "recommendation_config.json"

DEFAULT_CONFIG = {
    "data_dir": str(ROOT_DIR / "data"),
    "model_dir": str(ROOT_DIR / "algos"),
    "logs_dir": str(ROOT_DIR / "logs"),
    "train_test_split_ratio": 0.2,
    "max_features": 5000,
    "similarity_threshold": 0.3,
    "top_n_similar": 10,
    "use_cache": True,
    "cache_strategy": "lru",
    "lru_cache_max_size": 1500,
    "cache_ttl": 3600,
    "dimensionality_reduction": True,
    "n_components": 100,
    "mysql_host": MYSQL_HOST,
    "mysql_name": MYSQL_NAME,
    "mysql_user": MYSQL_USER,
    "mysql_password": MYSQL_PASSWORD,
    "redis_host": REDIS_HOST,
    "redis_port": REDIS_PORT,
    "redis_password": REDIS_PASSWORD
}

class RecommendationConfig:
    """Manages configuration for the recommendation pipeline."""

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load config from file or initialize with defaults if file doesn't exist."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info("Configuration loaded", config_path=str(self.config_path))
            else:
                logger.warning("Config file not found. Initializing with defaults.")
                self.config = DEFAULT_CONFIG.copy()
                self.save_config(self.config)

            self._apply_config()

        except Exception as e:
            logger.error(f"Failed to load or initialize config: {e}")
            self.config = DEFAULT_CONFIG.copy()
            self._apply_config()

    def _apply_config(self) -> None:
        """Assign config values to instance attributes."""
        self.data_dir = Path(self.config["data_dir"])
        self.model_dir = Path(self.config["model_dir"])
        self.logs_dir = Path(self.config["logs_dir"])
        self.train_test_split_ratio = self.config["train_test_split_ratio"]
        self.text_weight = self.config["text_weight"]
        self.max_features = self.config["max_features"]
        self.default_top_n = self.config["default_top_n"]
        self.use_cache = self.config["use_cache"]
        self.cache_ttl = self.config["cache_ttl"]
        self.retraining_frequency = self.config["retraining_frequency"]
        self.retraining_time = self.config["retraining_time"]
        self.api_host = self.config["api_host"]
        self.api_port = self.config["api_port"]
        self.metrics_port = self.config["metrics_port"]

        # Ensure required directories exist
        for directory in [self.data_dir, self.model_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

    def save_config(self, config_dict: Dict[str, Any]) -> None:
        """Save the given configuration dictionary to the config file."""
        try:
            os.makedirs(self.config_path.parent, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)
            logger.info("Configuration saved", config_path=str(self.config_path))
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def get_config_dict(self) -> Dict[str, Any]:
        """Return the current configuration as a dictionary."""
        return self.config.copy()