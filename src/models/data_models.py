"""
Pydantic models for data validation and serialization.
These models define the structure of the requests and responses for our API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any


class ProductEmbedding(BaseModel):
    """Model representing a product with its embedding vector."""
    product_id: str
    embedding: List[float]
    category: Optional[str] = None
    gender: Optional[str] = None
    title: Optional[str] = None
    image_url: Optional[str] = None


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch processing of embeddings."""
    products: List[ProductEmbedding]
    pca_components: int = Field(default=50, ge=2, le=100)
    final_components: int = Field(default=2, ge=2, le=3)
    use_precomputed_pca: bool = False


class VisualizationPoint(BaseModel):
    """Model representing a product in the visualization space."""
    product_id: str
    x: float
    y: float
    z: Optional[float] = None  # Only present for 3D visualizations
    category: Optional[str] = None
    gender: Optional[str] = None
    title: Optional[str] = None
    image_url: Optional[str] = None


class VisualizationResponse(BaseModel):
    """Response model for visualization API."""
    points: List[VisualizationPoint]
    processing_time_ms: float


class ErrorResponse(BaseModel):
    """Model for API error responses."""
    error: str
    details: Optional[str] = None


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str = "ok"
    version: str = "1.0.0"