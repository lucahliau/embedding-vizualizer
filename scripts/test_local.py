"""
Test script to verify the dimensionality reduction functionality locally.
This script generates random embeddings and processes them using our service.
"""

import sys
import os
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to sys.path to allow importing our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dimension_reducer import DimensionReducer
from src.models.data_models import ProductEmbedding, BatchEmbeddingRequest
from src.api.routes import visualize_embeddings

def generate_sample_embeddings(num_samples=100, dimension=896):
    """Generate random embeddings for testing."""
    print(f"Generating {num_samples} sample embeddings of dimension {dimension}...")
    
    # Categories and genders for sample data
    categories = ["clothing", "footwear", "accessories"]
    genders = ["male", "female", "unisex"]
    
    # Generate random embeddings
    products = []
    for i in range(num_samples):
        # Create random embedding vector
        embedding = np.random.randn(dimension).tolist()
        
        # Assign random category and gender
        category = categories[i % len(categories)]
        gender = genders[i % len(genders)]
        
        # Create product with embedding
        product = ProductEmbedding(
            product_id=f"test-product-{i}",
            embedding=embedding,
            category=category,
            gender=gender,
            title=f"Test Product {i}",
            image_url=f"https://example.com/images/product-{i}.jpg"
        )
        products.append(product)
    
    return products

def test_dimension_reducer_directly():
    """Test the DimensionReducer class directly."""
    print("\n--- Testing DimensionReducer directly ---")
    
    # Generate sample embeddings (as numpy array)
    num_samples = 100
    dimension = 896
    embeddings = np.random.randn(num_samples, dimension)
    
    # Create dimension reducer
    reducer = DimensionReducer()
    
    # Test PCA reduction
    start_time = time.time()
    reduced_embeddings = reducer.fit_pca(embeddings, n_components=50)
    pca_time = time.time() - start_time
    
    print(f"PCA reduction completed in {pca_time:.2f} seconds")
    print(f"Reduced shape: {reduced_embeddings.shape}")
    
    # Test t-SNE visualization
    start_time = time.time()
    viz_coords = reducer.fit_transform_tsne(reduced_embeddings, n_components=2)
    tsne_time = time.time() - start_time
    
    print(f"t-SNE completed in {tsne_time:.2f} seconds")
    print(f"Visualization shape: {viz_coords.shape}")
    
    # Test full pipeline
    start_time = time.time()
    viz_coords = reducer.process_embeddings(embeddings, 
                                           pca_components=50, 
                                           final_components=2)
    pipeline_time = time.time() - start_time
    
    print(f"Full pipeline completed in {pipeline_time:.2f} seconds")
    
    return viz_coords

async def test_api_endpoint():
    """Test the API endpoint using the request model."""
    print("\n--- Testing API endpoint ---")
    
    # Generate sample products
    products = generate_sample_embeddings(num_samples=100, dimension=896)
    
    # Create request
    request = BatchEmbeddingRequest(
        products=products,
        pca_components=50,
        final_components=2
    )
    
    # Call the endpoint
    print("Calling visualize_embeddings endpoint...")
    start_time = time.time()
    response = await visualize_embeddings(request)
    api_time = time.time() - start_time
    
    print(f"API endpoint completed in {api_time:.2f} seconds")
    print(f"Returned {len(response.points)} visualization points")
    
    return response

def plot_visualization(viz_coords, categories=None, output_file="visualization_test.png"):
    """Plot the visualization result."""
    plt.figure(figsize=(10, 8))
    
    if categories is None:
        # Simple scatterplot
        plt.scatter(viz_coords[:, 0], viz_coords[:, 1])
    else:
        # Color by category
        unique_categories = list(set(categories))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
        
        for i, category in enumerate(unique_categories):
            mask = [c == category for c in categories]
            plt.scatter(
                viz_coords[mask, 0], 
                viz_coords[mask, 1],
                label=category,
                color=colors[i]
            )
        
        plt.legend()
    
    plt.title("Sample Embedding Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

async def main():
    """Run all tests."""
    print("=== EMBEDDING VISUALIZATION SERVICE TEST ===")
    
    # Test dimension reducer directly
    viz_coords = test_dimension_reducer_directly()
    
    # Test API endpoint
    response = await test_api_endpoint()
    
    # Extract coordinates and categories from response
    api_coords = np.array([[point.x, point.y] for point in response.points])
    categories = [point.category for point in response.points]
    
    # Plot results
    print("\n--- Plotting results ---")
    plot_visualization(api_coords, categories, "api_visualization_test.png")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())