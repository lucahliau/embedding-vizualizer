"""
API routes for the embedding visualization service.
Defines endpoints for dimensionality reduction and health checks.
"""

import time
import logging
import numpy as np
from fastapi import APIRouter, HTTPException, File, UploadFile, Query, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import json
import random
from typing import Optional, List, Dict, Any

from src.models.data_models import (
    BatchEmbeddingRequest, 
    VisualizationResponse, 
    VisualizationPoint,
    HealthResponse,
    ProductEmbedding
)
from src.models.dimension_reducer import DimensionReducer
from src.config.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Create a shared dimension reducer instance
dimension_reducer = DimensionReducer()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}

@router.get("/categories")
async def get_categories():
    """Get all available product categories from the database."""
    try:
        # Import here to avoid circular imports
        from src.db.mongodb import get_product_categories
        
        # Get categories from MongoDB
        categories = await get_product_categories()
        
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error fetching categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualize", response_model=VisualizationResponse)
async def visualize_embeddings(request: BatchEmbeddingRequest):
    """
    Process a batch of embeddings and return visualization coordinates.
    
    This endpoint takes a batch of product embeddings and returns
    2D or 3D coordinates for visualization.
    """
    try:
        start_time = time.time()
        
        # Extract embeddings and metadata
        products = request.products
        
        if not products:
            raise HTTPException(status_code=400, detail="No products provided")
            
        # Collect embeddings into a numpy array
        embeddings = np.array([product.embedding for product in products])
        
        logger.info(f"Processing batch of {len(products)} products")
        
        # Apply dimensionality reduction
        viz_coords = dimension_reducer.process_embeddings(
            embeddings,
            pca_components=request.pca_components,
            final_components=request.final_components,
            use_precomputed_pca=request.use_precomputed_pca
        )
        
        # Create visualization points
        points = []
        for i, product in enumerate(products):
            point = {
                "product_id": product.product_id,
                "x": float(viz_coords[i, 0]),
                "y": float(viz_coords[i, 1]),
                "category": product.category,
                "gender": product.gender,
                "title": product.title,
                "image_url": product.image_url
            }
            
            # Add z-coordinate for 3D visualizations
            if request.final_components == 3:
                point["z"] = float(viz_coords[i, 2])
                
            points.append(VisualizationPoint(**point))
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return VisualizationResponse(
            points=points,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualize/file")
async def visualize_from_file(
    file: UploadFile = File(...),
    pca_components: int = 50,
    final_components: int = 2
):
    """
    Process embeddings from a CSV or JSON file and return visualization coordinates.
    
    This allows uploading a file with embedding data instead of sending it in the request body.
    """
    try:
        start_time = time.time()
        
        # Read the file content
        content = await file.read()
        
        # Determine file type and parse accordingly
        if file.filename.endswith('.csv'):
            # Parse CSV
            df = pd.read_csv(pd.io.common.BytesIO(content))
            
        elif file.filename.endswith('.json'):
            # Parse JSON
            data = json.loads(content)
            df = pd.DataFrame(data)
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload CSV or JSON."
            )
            
        # Validate the data format
        required_columns = ['product_id', 'embedding']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"File must contain these columns: {', '.join(required_columns)}"
            )
            
        # Process the embeddings
        products = []
        for _, row in df.iterrows():
            # Handle embedding format differences (string vs list)
            embedding = row['embedding']
            if isinstance(embedding, str):
                embedding = json.loads(embedding.replace("'", '"'))
                
            product = {
                "product_id": str(row['product_id']),
                "embedding": embedding,
            }
            
            # Add optional fields if present
            for field in ['category', 'gender', 'title', 'image_url']:
                if field in df.columns:
                    product[field] = row[field]
            
            products.append(product)
            
        # Create request object
        request = BatchEmbeddingRequest(
            products=products,
            pca_components=pca_components,
            final_components=final_components
        )
        
        # Process using the existing endpoint logic
        return await visualize_embeddings(request)
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/filter")
async def get_filtered_visualization(
    category: str = Query(..., description="Product category (clothing, footwear, accessories)"),
    gender: Optional[str] = Query(None, description="Gender filter (male, female, unisex)"),
    limit: int = Query(1000, ge=100, le=10000, description="Maximum number of products to include"),
    pca_components: int = Query(50, ge=10, le=100, description="PCA components to use"),
    final_components: int = Query(2, ge=2, le=3, description="Final visualization dimensions")
):
    """
    Generate visualization for products filtered by category and gender.
    
    This endpoint retrieves filtered products from MongoDB and processes them.
    """
    try:
        # Log request information
        logger.info(f"Filter request: category={category}, gender={gender}, limit={limit}")
        
        # Import the mongodb module here to avoid circular imports
        from src.db.mongodb import get_product_embeddings
        
        # Get products from MongoDB
        products_data = await get_product_embeddings(category, gender, limit)
        
        if not products_data:
            return JSONResponse(content={
                "message": "No products found matching the criteria",
                "points": []
            })
        
        # Convert to our data model
        products = []
        for product in products_data:
            try:
                # Extract product fields
                product_id = product.get("product_id")
                embedding = product.get("embedding")
                
                if not product_id or not embedding:
                    continue
                
                # Create product with embedding
                products.append({
                    "product_id": product_id,
                    "embedding": embedding,
                    "category": product.get("category"),
                    "gender": product.get("gender"),
                    "title": product.get("title"),
                    "image_url": product.get("imageUrl")
                })
            except Exception as e:
                logger.warning(f"Error processing product {product.get('product_id')}: {str(e)}")
                continue
        
        # Create request object
        request = BatchEmbeddingRequest(
            products=products,
            pca_components=pca_components,
            final_components=final_components
        )
        
        # Process using the existing endpoint logic
        return await visualize_embeddings(request)
        
    except Exception as e:
        logger.error(f"Error generating filtered visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/demo")
async def get_demo_visualization():
    """
    Generate a small demo visualization dataset.
    
    This endpoint creates some sample data for initial visualization.
    """
    try:
        # Generate a small set of demo products across categories
        categories = ["clothing", "footwear", "accessories"]
        products = []
        
        for category in categories:
            products.extend(generate_demo_products(category, None, 100))
        
        # Create request with smaller PCA dimensions for speed
        request = BatchEmbeddingRequest(
            products=products,
            pca_components=20,
            final_components=2
        )
        
        # Process using the existing endpoint logic
        return await visualize_embeddings(request)
        
    except Exception as e:
        logger.error(f"Error generating demo visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_demo_products(category, gender=None, count=100):
    """
    Generate random product data for demonstration purposes.
    
    In a real implementation, this would be replaced with database queries.
    """
    products = []
    
    # Set up constants
    genders = ["male", "female", "unisex"] if gender is None else [gender]
    dimension = 896  # The high-dimensional embedding
    
    # Create category-specific patterns to simulate clusters
    category_base = {
        "clothing": np.random.randn(dimension) * 0.5 + np.array([1.0] * dimension),
        "footwear": np.random.randn(dimension) * 0.5 + np.array([-1.0] * dimension),
        "accessories": np.random.randn(dimension) * 0.5 + np.array([0.0, 0.5] * (dimension // 2))
    }
    
    gender_mod = {
        "male": np.array([0.2] * dimension),
        "female": np.array([-0.2] * dimension),
        "unisex": np.array([0.0] * dimension)
    }
    
    # Generate products with similar embeddings based on category/gender
    for i in range(count):
        # Select random gender
        gender_value = random.choice(genders)
        
        # Generate embedding with category and gender patterns
        base_embedding = category_base[category]
        gender_modifier = gender_mod[gender_value]
        
        # Add random noise to create variations
        noise = np.random.randn(dimension) * 0.2
        
        # Combine base, gender modifier, and noise
        embedding = base_embedding + gender_modifier + noise
        
        # Create product
        product = {
            "product_id": f"demo-{category}-{gender_value}-{i}",
            "embedding": embedding.tolist(),
            "category": category,
            "gender": gender_value,
            "title": f"{gender_value.capitalize()} {category} Item {i}",
            "image_url": f"https://via.placeholder.com/200?text={category}+{i}"
        }
        
        products.append(product)
    
    return products