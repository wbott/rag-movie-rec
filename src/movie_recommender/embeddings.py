"""
Embedding providers for the movie recommender system.

Supports both OpenAI embeddings (API-based, no local ML libraries needed)
and Sentence Transformers (local models, requires ML libraries).
"""

import numpy as np
from typing import List
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """Encode texts into embeddings."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API-based embedding provider (no local ML libraries needed)."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        """
        Initialize OpenAI embedding provider.
        
        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings")
        
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self._dimension = self._get_dimension()
    
    def _get_dimension(self) -> int:
        """Get embedding dimension for the model."""
        # Common OpenAI embedding model dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model_name, 1536)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode texts using OpenAI embeddings API.
        
        Args:
            texts: List of texts to encode
            show_progress_bar: Whether to show progress (ignored for API calls)
            
        Returns:
            Array of embeddings
        """
        if show_progress_bar and len(texts) > 10:
            print(f"Encoding {len(texts)} texts using OpenAI API...")
        
        try:
            # OpenAI API can handle multiple texts at once
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            # Extract embeddings
            embeddings = []
            for data in response.data:
                embeddings.append(data.embedding)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Failed to get embeddings from OpenAI: {e}")


class SentenceTransformersProvider(EmbeddingProvider):
    """Sentence Transformers local embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformers provider.
        
        Args:
            model_name: Sentence transformer model name
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
    
    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode texts using Sentence Transformers.
        
        Args:
            texts: List of texts to encode
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


def get_embedding_provider(provider_type: str = "openai", **kwargs) -> EmbeddingProvider:
    """
    Get an embedding provider instance.
    
    Args:
        provider_type: Type of provider ("openai" or "sentence_transformers")
        **kwargs: Provider-specific arguments
        
    Returns:
        EmbeddingProvider instance
    """
    if provider_type.lower() == "openai":
        return OpenAIEmbeddingProvider(**kwargs)
    elif provider_type.lower() in ["sentence_transformers", "local"]:
        return SentenceTransformersProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")