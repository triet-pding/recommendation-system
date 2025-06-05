from pathlib import Path
import structlog
import os
import json
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
from dataclasses import dataclass

logger = structlog.get_logger()

@dataclass
class DatabaseConfig:
    """Database configuration container."""
    mysql_host: str
    mysql_name: str
    mysql_user: str
    mysql_password: str
    redis_host: str
    redis_port: int
    redis_password: Optional[str] = None


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class RecommendationConfig:
    """
    Manages configuration for the recommendation pipeline.
    
    Provides centralized configuration management with validation,
    environment variable integration, and proper error handling.
    """

    # Default configuration template
    DEFAULT_CONFIG = {
        "data_dir": "data",
        "model_dir": "models", 
        "logs_dir": "logs",
        "train_test_split_ratio": 0.2,
        "max_features": 5000,
        "similarity_threshold": 0.3,
        "top_n_similar": 10,
        "use_cache": True,
        "cache_strategy": "lru",
        "lru_cache_max_size": 1500,
        "min_interactions": 3,
        "cache_ttl": 3600,
        "dimensionality_reduction": True,
        "n_components": 100
    }

    # Configuration validation rules
    VALIDATION_RULES = {
        "train_test_split_ratio": {"type": (int, float), "min": 0.0, "max": 1.0},
        "max_features": {"type": int, "min": 1},
        "similarity_threshold": {"type": (int, float), "min": 0.0, "max": 1.0},
        "top_n_similar": {"type": int, "min": 1},
        "lru_cache_max_size": {"type": int, "min": 1},
        "min_interactions": {"type": int, "min": 1},
        "cache_ttl": {"type": int, "min": 0},
        "n_components": {"type": int, "min": 1}
    }

    def __init__(self, config_path: Optional[Union[str, Path]] = None, 
                 auto_create_dirs: bool = True) -> None:
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
            auto_create_dirs: Whether to automatically create required directories.
        """
        # Load environment variables
        load_dotenv()
        
        # Set paths
        self.root_dir = Path(__file__).resolve().parent
        self.config_path = Path(config_path) if config_path else (
            self.root_dir / "config" / "recommendation_config.json"
        )
        
        # Initialize configuration
        self.config: Dict[str, Any] = {}
        self._db_config: Optional[DatabaseConfig] = None
        self.auto_create_dirs = auto_create_dirs
        
        # Load and validate configuration
        self._load_configuration()
        self._load_database_config()
        
        if self.auto_create_dirs:
            self._create_directories()

    def _load_configuration(self) -> None:
        """Load and validate configuration from file or initialize with defaults."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                self.config = {**self.DEFAULT_CONFIG, **loaded_config}
                logger.info("Configuration loaded successfully", 
                          config_path=str(self.config_path))
            else:
                logger.info("Config file not found, using defaults", 
                          config_path=str(self.config_path))
                self.config = self.DEFAULT_CONFIG.copy()
                self.save_config()

            # Validate configuration
            self._validate_config()
            
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in config file", error=str(e))
            raise ConfigurationError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise ConfigurationError(f"Configuration loading failed: {e}")

    def _load_database_config(self) -> None:
        """Load database configuration from environment variables."""
        try:
            # Required environment variables
            required_vars = {
                'MYSQL_DB_HOST': 'mysql_host',
                'MYSQL_DB_NAME': 'mysql_name', 
                'MYSQL_DB_USER': 'mysql_user',
                'MYSQL_DB_PASSWORD': 'mysql_password',
                'REDIS_DB_HOST': 'redis_host',
                'REDIS_DB_PORT': 'redis_port'
            }
            
            env_values = {}
            missing_vars = []
            
            for env_var, config_key in required_vars.items():
                value = os.getenv(env_var)
                if value is None:
                    missing_vars.append(env_var)
                else:
                    env_values[config_key] = value
            
            if missing_vars:
                raise ConfigurationError(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )
            
            # Handle Redis port conversion
            try:
                env_values['redis_port'] = int(env_values['redis_port'])
            except (ValueError, TypeError):
                raise ConfigurationError("REDIS_DB_PORT must be a valid integer")
            
            # Optional Redis password
            env_values['redis_password'] = os.getenv('REDIS_DB_PASSWORD')
            
            self._db_config = DatabaseConfig(**env_values)
            logger.info("Database configuration loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load database configuration", error=str(e))
            raise ConfigurationError(f"Database configuration failed: {e}")

    def _validate_config(self) -> None:
        """Validate configuration values against defined rules."""
        errors = []
        
        for key, rules in self.VALIDATION_RULES.items():
            if key not in self.config:
                continue
                
            value = self.config[key]
            
            # Type validation
            expected_type = rules["type"]
            if not isinstance(value, expected_type):
                errors.append(f"{key}: expected {expected_type}, got {type(value)}")
                continue
            
            # Range validation
            if "min" in rules and value < rules["min"]:
                errors.append(f"{key}: value {value} below minimum {rules['min']}")
            
            if "max" in rules and value > rules["max"]:
                errors.append(f"{key}: value {value} above maximum {rules['max']}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            logger.error("Configuration validation failed", errors=errors)
            raise ConfigurationError(error_msg)

    def _create_directories(self) -> None:
        """Create required directories if they don't exist."""
        directories = [
            self.data_dir,
            self.model_dir, 
            self.logs_dir,
            self.config_path.parent
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info("Directory created/verified", path=str(directory))
            except Exception as e:
                logger.warning("Failed to create directory", 
                             path=str(directory), error=str(e))

    def save_config(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_dict: Configuration to save. If None, saves current config.
        """
        config_to_save = config_dict if config_dict is not None else self.config
        
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4, sort_keys=True)
            
            logger.info("Configuration saved successfully", 
                       config_path=str(self.config_path))
            
        except Exception as e:
            logger.error("Failed to save configuration", error=str(e))
            raise ConfigurationError(f"Configuration save failed: {e}")

    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates.
            validate: Whether to validate the updated configuration.
        """
        old_config = self.config.copy()
        
        try:
            # Apply updates
            self.config.update(updates)
            
            # Validate if requested
            if validate:
                self._validate_config()
            
            # Save updated configuration
            self.save_config()
            
            logger.info("Configuration updated successfully", updates=list(updates.keys()))
            
        except Exception as e:
            # Rollback on failure
            self.config = old_config
            logger.error("Configuration update failed, rolled back", error=str(e))
            raise

    def get_config_dict(self) -> Dict[str, Any]:
        """Return a copy of the current configuration dictionary."""
        return self.config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key to retrieve.
            default: Default value if key doesn't exist.
            
        Returns:
            Configuration value or default.
        """
        return self.config.get(key, default)

    def reload_config(self) -> None:
        """Reload configuration from file."""
        logger.info("Reloading configuration")
        self._load_configuration()
        if self.auto_create_dirs:
            self._create_directories()

    # Property accessors for common configuration values
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        path = self.config.get("data_dir", "data")
        return self.root_dir / path if not Path(path).is_absolute() else Path(path)

    @property
    def model_dir(self) -> Path:
        """Get model directory path."""
        path = self.config.get("model_dir", "models")
        return self.root_dir / path if not Path(path).is_absolute() else Path(path)

    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        path = self.config.get("logs_dir", "logs")
        return self.root_dir / path if not Path(path).is_absolute() else Path(path)

    @property
    def database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        if self._db_config is None:
            raise ConfigurationError("Database configuration not loaded")
        return self._db_config

    # Convenience properties for database access
    @property
    def mysql_host(self) -> str:
        return self.database_config.mysql_host

    @property
    def mysql_name(self) -> str:
        return self.database_config.mysql_name

    @property
    def mysql_user(self) -> str:
        return self.database_config.mysql_user

    @property
    def mysql_password(self) -> str:
        return self.database_config.mysql_password

    @property
    def redis_host(self) -> str:
        return self.database_config.redis_host

    @property
    def redis_port(self) -> int:
        return self.database_config.redis_port

    @property
    def redis_password(self) -> Optional[str]:
        return self.database_config.redis_password

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"RecommendationConfig(config_path='{self.config_path}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"RecommendationConfig with {len(self.config)} settings loaded from {self.config_path}"