"""
Module for performing dimensionality reduction on embeddings.
This handles the reduction from high-dimensional embeddings (896D)
to lower dimensions suitable for visualization (2D/3D).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DimensionReducer:
    """Class to handle dimensionality reduction of embeddings."""
    
    def __init__(self):
        """Initialize the dimension reducer."""
        # PCA for initial reduction (896D -> 50D)
        self.pca = None
        # t-SNE for final visualization (50D -> 2D)
        self.tsne = None
        
    def fit_pca(self, embeddings, n_components=50):
        """
        Fit PCA model on the input embeddings.
        
        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            n_components: number of components to reduce to
            
        Returns:
            Reduced embeddings of shape (n_samples, n_components)
        """
        logger.info(f"Fitting PCA to reduce to {n_components} dimensions")
        start_time = time.time()
        
        # Check input shape
        if len(embeddings.shape) != 2:
            raise ValueError(f"Expected 2D array, got {len(embeddings.shape)}D")
            
        # Initialize and fit PCA
        self.pca = PCA(n_components=n_components)
        reduced_embeddings = self.pca.fit_transform(embeddings)
        
        elapsed_time = time.time() - start_time
        logger.info(f"PCA completed in {elapsed_time:.2f} seconds")
        logger.info(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        return reduced_embeddings
        
    def transform_pca(self, embeddings):
        """
        Transform embeddings using pre-fitted PCA model.
        
        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            
        Returns:
            Reduced embeddings
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit_pca first.")
        
        logger.info("Transforming embeddings with fitted PCA")
        return self.pca.transform(embeddings)
        
    def fit_transform_tsne(self, embeddings, n_components=2, perplexity=30):
        """
        Apply t-SNE to embeddings for visualization.
        
        Args:
            embeddings: numpy array (typically output from PCA)
            n_components: dimensions for visualization (2 or 3)
            perplexity: t-SNE perplexity parameter
            
        Returns:
            Visualization coordinates of shape (n_samples, n_components)
        """
        logger.info(f"Applying t-SNE to create {n_components}D visualization")
        start_time = time.time()
        
        # Initialize and fit t-SNE
        self.tsne = TSNE(n_components=n_components, 
                         perplexity=min(perplexity, embeddings.shape[0] - 1),
                         random_state=42)
        viz_coords = self.tsne.fit_transform(embeddings)
        
        elapsed_time = time.time() - start_time
        logger.info(f"t-SNE completed in {elapsed_time:.2f} seconds")
        
        return viz_coords
        
    def process_embeddings(self, embeddings, 
                          pca_components=50, 
                          final_components=2,
                          use_precomputed_pca=False):
        """
        Complete pipeline to process embeddings from high dimension to visualization.
        
        Args:
            embeddings: numpy array of original embeddings
            pca_components: intermediate dimension after PCA
            final_components: final dimension (2D or 3D)
            use_precomputed_pca: whether to use an already fitted PCA model
            
        Returns:
            Final coordinates for visualization
        """
        logger.info(f"Processing embeddings: {embeddings.shape[0]} samples, "
                   f"{embeddings.shape[1]} dimensions")
        
        # Step 1: PCA reduction
        if use_precomputed_pca and self.pca is not None:
            pca_result = self.transform_pca(embeddings)
        else:
            pca_result = self.fit_pca(embeddings, n_components=pca_components)
            
        # Step 2: t-SNE for visualization
        viz_coords = self.fit_transform_tsne(pca_result, n_components=final_components)
        
        logger.info("Embedding processing complete")
        return viz_coords