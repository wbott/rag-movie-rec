"""Tests for vector store module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from movie_recommender.vector_store import MovieVectorStore


class TestMovieVectorStore:
    """Test cases for MovieVectorStore class."""
    
    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        store = MovieVectorStore()
        
        assert store.embedding_model is not None
        assert store.index is None
        assert store.chunks_df is None
        assert store.dimension is None
    
    def test_create_embeddings(self, mock_vector_store):
        """Test embedding creation."""
        texts = ["This is a test movie", "Another test movie"]
        embeddings = mock_vector_store.create_embeddings(texts, show_progress=False)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Should have some dimensions
        assert embeddings.dtype == np.float32
    
    def test_build_index(self, mock_vector_store, sample_descriptions):
        """Test building FAISS index."""
        # Create chunks from descriptions
        chunks_data = []
        for _, row in sample_descriptions.iterrows():
            chunks_data.append({
                'Title': row['Title'],
                'Chunk': row['Description'],
                'Metadata': {'Title': row['Title']}
            })
        
        chunks_df = pd.DataFrame(chunks_data)
        
        mock_vector_store.build_index(chunks_df)
        
        assert mock_vector_store.index is not None
        assert mock_vector_store.chunks_df is not None
        assert mock_vector_store.dimension > 0
        assert mock_vector_store.index.ntotal == 3
    
    def test_search_functionality(self, mock_vector_store, sample_descriptions):
        """Test search functionality."""
        # Build index first
        chunks_data = []
        for _, row in sample_descriptions.iterrows():
            chunks_data.append({
                'Title': row['Title'],
                'Chunk': row['Description'],
                'Metadata': {'Title': row['Title']}
            })
        
        chunks_df = pd.DataFrame(chunks_data)
        mock_vector_store.build_index(chunks_df)
        
        # Test search
        results = mock_vector_store.search("action movie", k=2)
        
        assert len(results) == 2
        assert all('Title' in result for result in results)
        assert all('Chunk' in result for result in results)
        assert all('Metadata' in result for result in results)
        assert all('Distance' in result for result in results)
    
    def test_save_and_load_index(self, mock_vector_store, sample_descriptions, temp_dir):
        """Test saving and loading index."""
        # Build index
        chunks_data = []
        for _, row in sample_descriptions.iterrows():
            chunks_data.append({
                'Title': row['Title'],
                'Chunk': row['Description'],
                'Metadata': {'Title': row['Title']}
            })
        
        chunks_df = pd.DataFrame(chunks_data)
        mock_vector_store.build_index(chunks_df)
        
        # Save index
        index_path = os.path.join(temp_dir, "test_index.index")
        metadata_path = os.path.join(temp_dir, "test_metadata.csv")
        
        mock_vector_store.save_index(index_path, metadata_path)
        
        assert os.path.exists(index_path)
        assert os.path.exists(metadata_path)
        
        # Create new store and load
        new_store = MovieVectorStore()
        new_store.load_index(index_path, metadata_path)
        
        assert new_store.index is not None
        assert new_store.chunks_df is not None
        assert new_store.dimension == mock_vector_store.dimension
        assert new_store.index.ntotal == mock_vector_store.index.ntotal
    
    def test_get_stats(self, mock_vector_store, sample_descriptions):
        """Test getting statistics."""
        # Test without loading
        stats = mock_vector_store.get_stats()
        assert stats["status"] == "not_loaded"
        
        # Build index
        chunks_data = []
        for _, row in sample_descriptions.iterrows():
            chunks_data.append({
                'Title': row['Title'],
                'Chunk': row['Description'],
                'Metadata': {'Title': row['Title']}
            })
        
        chunks_df = pd.DataFrame(chunks_data)
        mock_vector_store.build_index(chunks_df)
        
        # Test with loaded index
        stats = mock_vector_store.get_stats()
        assert stats["status"] == "loaded"
        assert stats["total_vectors"] == 3
        assert stats["unique_movies"] == 3
        assert stats["dimension"] > 0
    
    def test_search_without_index_raises_error(self, mock_vector_store):
        """Test that search raises error without index."""
        with pytest.raises(ValueError, match="Index not loaded"):
            mock_vector_store.search("test query")
    
    def test_save_without_index_raises_error(self, mock_vector_store, temp_dir):
        """Test that save raises error without index."""
        index_path = os.path.join(temp_dir, "test_index.index")
        metadata_path = os.path.join(temp_dir, "test_metadata.csv")
        
        with pytest.raises(ValueError, match="Index not built"):
            mock_vector_store.save_index(index_path, metadata_path)