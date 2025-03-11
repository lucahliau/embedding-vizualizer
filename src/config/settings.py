"""
Configuration settings for the Embedding Visualization Service.
Environment variables can override these defaults.
"""

import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings loaded from environment variables with defaults."""
    
    # Application settings
    APP_NAME: str = "Embedding Visualization Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server settings
    HOST: str = "0.0.0.0"  # Use 0.0.0.0 for production, 127.0.0.1 for local development
    PORT: int = 8000
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]  # Allow all origins in development
    
    # Dimensionality reduction defaults
    DEFAULT_PCA_COMPONENTS: int = 50
    DEFAULT_FINAL_COMPONENTS: int = 2
    MAX_ITEMS_PER_REQUEST: int = 10000
    
    # API path prefix
    API_PREFIX: str = "/api"
    
    # MongoDB settings
    MONGODB_URI: str
    MONGODB_DB: str = ""  # Optional, can be extracted from URI
    MONGODB_COLLECTION: str = "posts"
    
    # Cloud storage settings (for later use with AWS)
    USE_CLOUD_STORAGE: bool = False
    CLOUD_STORAGE_BUCKET: str = ""
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        """Pydantic config for the settings."""
        case_sensitive = True
        env_file = ".env"  # Load from .env file if present

# Create a global settings object
settings = Settings()