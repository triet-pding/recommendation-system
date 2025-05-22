from config import RecommendationConfig
from typing import Optional, Any
import structlog
import hashlib
import pickle

logger = structlog.get_logger()

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class CacheManager:
    """Unified cache manager supporting multiple cache strategies"""
    def __init__(self, config: RecommendationConfig):
        self.config = config
        self.strategy = self.config.get('cache_strategy', 'lru')
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache = None
        self._setup_cache()

    def _setup_cache(self):
        """Setup cache based on strategy"""
        if self.strategy == 'redis' and REDIS_AVAILABLE:
            self._setup_redis_cache()
        elif self.strategy == 'lru':
            self._setup_lru_cache()
        else:
            logger.warning(f"Cache strategy '{self.strategy}' not available, using LRU")
            self.strategy = 'lru'
            self._setup_lru_cache()

    def _setup_redis_cache(self):
        """Setup Redis cache"""
        try:
            host = self.config.get('redis_host', 'localhost')
            port = self.config.get('redis_port', 6379)
            password = self.config.get('redis_password', 0)
            
            self.cache = redis.Redis(
                host=host, 
                port=port, 
                password=password,
                decode_responses=False,  # We'll handle binary data
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.cache.ping()
            
            # Configure LRU eviction policy if possible
            try:
                self.cache.config_set('maxmemory-policy', 'allkeys-lru')
            except redis.ResponseError:
                logger.warning("Could not set Redis maxmemory-policy")
            
            logger.info(f"Redis cache configured at {host}:{port}/")
            
        except Exception as e:
            logger.warning(f"Redis setup failed: {e}. Falling back to LRU cache")
            self.strategy = 'lru'
            self._setup_lru_cache()

    def _setup_lru_cache(self):
        """Setup in-memory LRU cache"""
        max_size = self.config.get('lru_cache_max_size', 1000)
        self.cache = {}
        self.max_size = max_size
        self.access_order = []  # Track access order for LRU
        logger.info(f"LRU cache configured with max_size={max_size}")

    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key"""
        # Create a string representation of all arguments
        key_parts = [prefix] + [str(arg) for arg in args]
        
        # Add sorted kwargs
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
        
        # Create hash if key is too long
        key_str = ":".join(key_parts)
        if len(key_str) > 200:  # Redis key limit consideration
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            return f"{prefix}:{key_hash}"
        
        return key_str
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.strategy == 'redis' and self.cache:
                result = self.cache.get(key)
                if result is not None:
                    self.cache_hits += 1
                    return pickle.loads(result)
                else:
                    self.cache_misses += 1
                    return None
            
            elif self.strategy == 'lru':
                if key in self.cache:
                    # Update access order
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    self.cache_hits += 1
                    return self.cache[key]
                else:
                    self.cache_misses += 1
                    return None
        
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_misses += 1
            return None
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            if self.strategy == 'redis' and self.cache:
                serialized = pickle.dumps(value)
                if ttl:
                    return self.cache.setex(key, ttl, serialized)
                else:
                    return self.cache.set(key, serialized)
            
            elif self.strategy == 'lru':
                # Handle LRU eviction
                if len(self.cache) >= self.max_size and key not in self.cache:
                    # Remove least recently used item
                    lru_key = self.access_order.pop(0)
                    del self.cache[lru_key]
                
                self.cache[key] = value
                
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                return True
        
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
        
    