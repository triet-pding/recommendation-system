from algos.collaborative_filtering_based import ItemBasedCFRecommender
from algos.content_based import ContentBasedRecommender
from recommendation_config import RecommendationConfig
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import structlog
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import time

logger = structlog.get_logger()
recommender_service: Optional['HybridRecommendationService'] = None

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID for recommendations")
    n_recommendations: int = Field(default=30, ge=1, le=100, description="Number of recommendations to return")
    n_top_rated: int = Field(default=5, ge=1, le=20, description="Number of top-rated items to include")
    n_similar: int = Field(default=5, ge=1, le=20, description="Number of similar items per top-rated item")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        return v.strip()

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[str]
    total_count: int
    processing_time_ms: float
    metadata: Dict = Field(default_factory=dict)

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[str]
    total_count: int
    processing_time_ms: float
    metadata: Dict = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: bool
    version: str

class HybridRecommendationService:
    
    """Handles model training, evaluation, and persistence"""
    def __init__(self, config: RecommendationConfig) -> None:
        self.config = config
        self.cf_recommender = None
        self.cbf_recommender = None
        self.models_loaded = False
        self._recommendation_cache = {}
        logger.info("Initializing HybridRecommendationService...")

    async def load_models(self) -> None:
        """Load models asynchronously"""
        try:
            # Load CF model
            await self._load_cf_model()
            logger.info("Collaborative Filtering recommender loaded")
            
            # Load CBF model
            await self._load_cbf_model()
            logger.info("Content-Based recommender loaded")
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
            raise

    async def _load_cf_model(self) -> None:
        """Load collaborative filtering model"""
        self.cf_recommender = ItemBasedCFRecommender(self.config)
        self.cf_recommender.load_model()
        
    async def _load_cbf_model(self) -> None:
        """Load content-based filtering model"""
        self.cbf_recommender = ContentBasedRecommender(self.config)
        self.cbf_recommender.load_models()

    def _validate_models(self) -> None:
        """Validate that models are loaded"""
        if not self.models_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are not loaded. Please try again later."
            )

    def _get_cache_key(self, user_id: str, n_recommendations: int, n_top_rated: int, n_similar: int) -> str:
        """Generate cache key for recommendations"""
        return f"{user_id}:{n_recommendations}:{n_top_rated}:{n_similar}"

    def _get_cached_recommendations(self, cache_key: str) -> Optional[List[str]]:
        """Get recommendations from cache if available and not expired"""
        if cache_key in self._recommendation_cache:
            cached_data, timestamp = self._recommendation_cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                return cached_data
            else:
                # Remove expired cache entry
                del self._recommendation_cache[cache_key]
        return None

    def _cache_recommendations(self, cache_key: str, recommendations: List[str]) -> None:
        """Cache recommendations with timestamp"""
        self._recommendation_cache[cache_key] = (recommendations, time.time())

    async def get_recommendations_for_user(
        self, 
        user_id: str, 
        n_recommendations: int = 30,
        n_top_rated: int = 5, 
        n_similar: int = 5
    ) -> Tuple[List[str], Dict]:
        """
        Get hybrid recommendations for a user
        
        Args:
            user_id: User identifier
            n_recommendations: Total number of recommendations to return
            n_top_rated: Number of top-rated items to include
            n_similar: Number of similar items per top-rated item
            
        Returns:
            Tuple of (recommendations list, metadata dict)
        """
        start_time = time.time()
        
        try:
            # Validate models are loaded
            self._validate_models()
            
            # Check cache first
            cache_key = self._get_cache_key(user_id, n_recommendations, n_top_rated, n_similar)
            cached_recs = self._get_cached_recommendations(cache_key)
            
            if cached_recs:
                logger.info(f"Returning cached recommendations for user {user_id}")
                metadata = {
                    "source": "cache",
                    "cf_recommendations": 0,
                    "cbf_recommendations": 0
                }
                return cached_recs, metadata

            # Generate fresh recommendations
            final_recommendations = []
            seen_videos = set()
            metadata = {
                "source": "fresh",
                "cf_recommendations": 0,
                "cbf_recommendations": 0
            }

            # Phase 1: Get top-rated recommendations from CF
            try:
                top_rated_recs = self.cf_recommender.recommend_for_user(
                    user_id=user_id, n_recommendations=10
                )
                
                
                logger.info(f"CF generated {len(top_rated_recs)} recommendations")
                
                # Add top-rated videos
                for vid_data in top_rated_recs:
                    vid_id = vid_data[0] if isinstance(vid_data, tuple) else vid_data
                    
                    if vid_id not in seen_videos and len(final_recommendations) < n_top_rated:
                        final_recommendations.append(vid_id)
                        seen_videos.add(vid_id)
                        metadata["cf_recommendations"] += 1
                        
                        if len(final_recommendations) >= n_top_rated:
                            break
                            
            except Exception as e:
                logger.warning(f"CF recommendation failed for user {user_id}: {e}")
                # Continue with CBF only

            # Phase 2: Add similar videos using CBF
            if len(final_recommendations) < n_recommendations:
                try:
                    for vid_data in top_rated_recs[:n_top_rated]:
                        vid_id = vid_data[0] if isinstance(vid_data, tuple) else vid_data
                        
                        similar_vids = self.cbf_recommender.find_similar_videos(
                            video_id=vid_id, top_n=n_similar
                        )
                        
                        for similar_vid in similar_vids:
                            similar_vid_id = similar_vid[0] if isinstance(similar_vid, tuple) else similar_vid
                            
                            if (similar_vid_id not in seen_videos and 
                                len(final_recommendations) < n_recommendations):
                                final_recommendations.append(similar_vid_id)
                                seen_videos.add(similar_vid_id)
                                metadata["cbf_recommendations"] += 1
                                
                                if len(final_recommendations) >= n_recommendations:
                                    break
                        
                        if len(final_recommendations) >= n_recommendations:
                            break
                            
                except Exception as e:
                    logger.warning(f"CBF recommendation failed for user {user_id}: {e}")

            # Cache the results
            self._cache_recommendations(cache_key, final_recommendations)
            
            # Add timing metadata
            processing_time = (time.time() - start_time) * 1000
            metadata["processing_time_ms"] = processing_time
            
            logger.info(f"Generated {len(final_recommendations)} recommendations for user {user_id}")
            return final_recommendations, metadata

        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate recommendations: {str(e)}"
            )
                
    def clear_cache(self) -> int:
        """Clear recommendation cache"""
        cache_size = len(self._recommendation_cache)
        self._recommendation_cache.clear()
        logger.info(f"Cleared {cache_size} cached recommendations")
        return cache_size

async def get_recommender_service() -> HybridRecommendationService:
    """Dependency to get the recommender service"""
    if recommender_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is not available"
        )
    return recommender_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global recommender_service
    
    # Startup
    try:
        logger.info("Starting up recommendation service...")
        config = RecommendationConfig()
        recommender_service = HybridRecommendationService(config)
        await recommender_service.load_models()
        logger.info("Recommendation service initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize recommendation service: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down recommendation service...")
        if recommender_service:
            recommender_service.clear_cache()

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Recommender System API",
    description="Production-ready API for hybrid collaborative filtering and content-based recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations_for_user(
    user_id: str,
    n_recommendations: int = Query(default=30, ge=1, le=100, description="Number of recommendations"),
    n_top_rated: int = Query(default=5, ge=1, le=20, description="Number of top-rated items"),
    n_similar: int = Query(default=5, ge=1, le=20, description="Number of similar items per top-rated"),
    service: HybridRecommendationService = Depends(get_recommender_service)
):
    """Get recommendations for a specific user"""
    start_time = time.time()
    
    recommendations, metadata = await service.get_recommendations_for_user(
        user_id=user_id,
        n_recommendations=n_recommendations,
        n_top_rated=n_top_rated,
        n_similar=n_similar
    )
    
    processing_time = (time.time() - start_time) * 1000
    
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
        total_count=len(recommendations),
        processing_time_ms=processing_time,
        metadata=metadata
    )

@app.delete("/cache")
async def clear_cache(
    service: HybridRecommendationService = Depends(get_recommender_service)
):
    """Clear recommendation cache"""
    cleared_count = service.clear_cache()
    return {"message": f"Cleared {cleared_count} cached recommendations"}



@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import datetime
    
    # Check if service is available without raising an exception
    global recommender_service
    models_loaded = recommender_service is not None and recommender_service.models_loaded if recommender_service else False
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        timestamp=datetime.datetime.now().isoformat(),
        models_loaded=models_loaded,
        version="1.0.0"
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hybrid Recommendation System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# For local development, uncomment this code
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )