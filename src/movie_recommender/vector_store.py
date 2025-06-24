"""Vector store implementation using FAISS for similarity search."""

import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Optional
from .embeddings import EmbeddingProvider, get_embedding_provider


class MovieVectorStore:
    """FAISS-based vector store for movie similarity search."""
    
    def __init__(self, embedding_provider: EmbeddingProvider = None, **provider_kwargs):
        """
        Initialize vector store.
        
        Args:
            embedding_provider: Custom embedding provider instance
            **provider_kwargs: Arguments for default provider (openai, sentence_transformers, etc.)
        """
        if embedding_provider is None:
            # Default to OpenAI embeddings (no heavy ML libraries needed)
            self.embedding_provider = get_embedding_provider("openai", **provider_kwargs)
        else:
            self.embedding_provider = embedding_provider
            
        self.index: Optional[faiss.Index] = None
        self.chunks_df: Optional[pd.DataFrame] = None
        self.dimension: Optional[int] = None
    
    def create_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Create embeddings for a list of texts."""
        return self.embedding_provider.encode(texts, show_progress_bar=show_progress)
    
    def build_index(self, chunks_df: pd.DataFrame) -> None:
        """Build FAISS index from chunks DataFrame."""
        # Generate embeddings
        embeddings = self.create_embeddings(chunks_df['Chunk'].tolist())
        
        # Store chunks data
        self.chunks_df = chunks_df.copy()
        self.chunks_df['Embedding'] = embeddings.tolist()
        
        # Create FAISS index
        self.dimension = self.embedding_provider.dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """Save FAISS index and metadata to files."""
        if self.index is None or self.chunks_df is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        faiss.write_index(self.index, index_path)
        self.chunks_df[['Title', 'Chunk', 'Metadata']].to_csv(metadata_path, index=False)
    
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """Load FAISS index and metadata from files."""
        self.index = faiss.read_index(index_path)
        self.chunks_df = pd.read_csv(metadata_path)
        self.dimension = self.index.d
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar movie chunks."""
        if self.index is None or self.chunks_df is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        # Create query embedding
        query_embedding = self.create_embeddings([query], show_progress=False)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks_df):
                result = self.chunks_df.iloc[idx]
                metadata = eval(result['Metadata']) if isinstance(result['Metadata'], str) else result['Metadata']
                results.append({
                    'Title': result['Title'],
                    'Chunk': result['Chunk'],
                    'Metadata': metadata,
                    'Distance': float(distances[0][len(results)])
                })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if self.index is None or self.chunks_df is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_chunks": len(self.chunks_df),
            "unique_movies": self.chunks_df['Title'].nunique()
        }