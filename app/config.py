"""
Configuration management for Aviation Weather API
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
import logging
from logging.handlers import RotatingFileHandler
import sys
import json
from datetime import timedelta
from typing import Any

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    APP_NAME: str = "Aviation Weather API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    
    DATABASE_URL: str = "sqlite:///./aviation_weather.db"
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    USE_REDIS: bool = False
    
    METAR_CACHE_MINUTES: int = 15
    TAF_CACHE_MINUTES: int = 60
    FORECAST_CACHE_MINUTES: int = 30
    
    OPENMETEO_BASE_URL: str = "https://api.open-meteo.com/v1"
    FAA_WEATHER_BASE_URL: str = "https://aviationweather.gov/api/data"
    
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    AVWX_API_KEY: Optional[str] = None
    AVWX_BASE_URL: Optional[str] = None
    USE_AVWX: bool = False
    
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "aviation_weather.log"
    
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    
    DAYS_KEEP_METAR: int = 7
    DAYS_KEEP_TAF: int = 3
    DAYS_KEEP_FORECAST: int = 2
    DAYS_KEEP_LOGS: int = 30
    
    STATION_DATABASE_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

def setup_logging(settings: Settings):
    """Configure application logging"""
    
    logger = logging.getLogger("aviation_weather")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    if settings.LOG_FILE:
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10485760,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

class CacheManager:
    """Redis cache manager for weather data"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.redis_client = None
        
        if settings.USE_REDIS:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD,
                    decode_responses=True
                )
                self.redis_client.ping()
                print("✓ Redis cache connected")
            except Exception as e:
                print(f"✗ Redis connection failed: {e}")
                self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_minutes: int = 15):
        """Set value in cache with TTL"""
        if not self.redis_client:
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            self.redis_client.setex(
                key,
                timedelta(minutes=ttl_minutes),
                serialized
            )
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str):
        """Delete key from cache"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        if not self.redis_client:
            return False
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            print(f"Cache clear error: {e}")
            return False

if __name__ == "__main__":
    settings = get_settings()
    print(f"Database URL: {settings.DATABASE_URL}")
    print(f"Redis enabled: {settings.USE_REDIS}")
    print(f"Log level: {settings.LOG_LEVEL}")