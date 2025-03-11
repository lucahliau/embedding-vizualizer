"""
MongoDB connection and data access for the embedding visualization service.
This module handles connecting to MongoDB and retrieving product data.
"""

import logging
import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING
from typing import List, Dict, Any, Optional
import numpy as np

from src.config.settings import settings

# Set up logging
logger = logging.getLogger(__name__)

# MongoDB client instance
client = None
db = None

async def connect_to_mongodb():
    """Connect to MongoDB using settings from configuration."""
    global client, db
    
    if client is not None:
        return  # Already connected
    
    logger.info(f"Connecting to MongoDB...")
    
    try:
        # Create async MongoDB client
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
        
        # If MONGODB_DB is provided, use it, otherwise extract from URI or use default
        db_name = settings.MONGODB_DB
        if not db_name or db_name == "":
            # Try to extract from URI if it's included
            from urllib.parse import urlparse
            uri_path = urlparse(settings.MONGODB_URI).path
            if uri_path and uri_path.strip('/'):
                db_name = uri_path.strip('/')
            else:
                db_name = "fashion_recommender"  # Default if not specified
        
        db = client[db_name]
        
        # Log successful connection
        logger.info(f"Connected to MongoDB database '{db_name}'")
        
        # Test connection by listing collection names
        collections = await db.list_collection_names()
        logger.info(f"Available collections: {', '.join(collections)}")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise e

async def close_mongodb_connection():
    """Close MongoDB connection."""
    global client
    if client is not None:
        client.close()
        client = None
        logger.info("Closed MongoDB connection")

async def get_product_embeddings(
    category: Optional[str] = None,
    gender: Optional[str] = None, 
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Retrieve product embeddings from MongoDB with optional filtering.
    
    Args:
        category: Optional filter for product category
        gender: Optional filter for product gender
        limit: Maximum number of products to retrieve
        
    Returns:
        List of product documents with embeddings
    """
    if db is None:
        await connect_to_mongodb()
        
    # Build query filter
    query: Dict[str, Any] = {}
    
    if category:
        query["category"] = category
        
    if gender and gender != "all":
        query["gender"] = gender
        
    logger.info(f"Querying products with filter: {query}, limit: {limit}")
    
    # Retrieve products from the database
    cursor = db[settings.MONGODB_COLLECTION].find(query).limit(limit)
    
    # Convert to list and process
    products = []
    async for doc in cursor:
        # Ensure the doc has an embedding
        if "embedding" in doc and doc["embedding"]:
            # Convert embedding to the format we need
            embedding = doc["embedding"]
            
            # Convert ObjectId to string
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
                
            # Rename _id to product_id for consistency with our models
            doc["product_id"] = doc.pop("_id")
                
            products.append(doc)
    
    logger.info(f"Retrieved {len(products)} products with embeddings")
    return products

async def get_product_categories():
    """Get list of all product categories in the database."""
    if db is None:
        await connect_to_mongodb()
        
    categories = await db[settings.MONGODB_COLLECTION].distinct("category")
    return categories

async def get_product_by_id(product_id: str):
    """Get a single product by ID."""
    if db is None:
        await connect_to_mongodb()
        
    # Convert string ID to ObjectId if needed
    from bson.objectid import ObjectId
    try:
        # Check if it's a valid ObjectId
        if len(product_id) == 24:
            # Try to get by ObjectId
            product = await db[settings.MONGODB_COLLECTION].find_one({"_id": ObjectId(product_id)})
            if product:
                # Convert ObjectId to string
                product["_id"] = str(product["_id"])
                # Rename _id to product_id for consistency
                product["product_id"] = product.pop("_id")
                return product
                
    except Exception:
        pass
        
    # Try with string ID
    product = await db[settings.MONGODB_COLLECTION].find_one({"_id": product_id})
    if product:
        # Convert ObjectId to string if needed
        if isinstance(product["_id"], ObjectId):
            product["_id"] = str(product["_id"])
        # Rename _id to product_id for consistency
        product["product_id"] = product.pop("_id")
        
    return product