"""Tests for data processor module."""

import pytest
import pandas as pd
import tempfile
import os
from movie_recommender.data_processor import MovieDataProcessor


class TestMovieDataProcessor:
    """Test cases for MovieDataProcessor class."""
    
    def test_format_star_cast_short_list(self, data_processor):
        """Test star cast formatting with short list."""
        cast = "Actor One, Actor Two"
        result = data_processor.format_star_cast(cast)
        assert result == "Actor One, Actor Two"
    
    def test_format_star_cast_long_list(self, data_processor):
        """Test star cast formatting with long list."""
        cast = "Actor One, Actor Two, Actor Three, Actor Four, Actor Five"
        result = data_processor.format_star_cast(cast)
        assert result == "Actor One, Actor Two, Actor Three, and others"
    
    def test_generate_description(self, data_processor, sample_movie_data):
        """Test movie description generation."""
        row = sample_movie_data.iloc[0]  # Gladiator
        description = data_processor.generate_description(row)
        
        assert "Gladiator (2000)" in description
        assert "action, drama" in description
        assert "Ridley Scott" in description
        assert "Russell Crowe, Joaquin Phoenix" in description
        assert "8.5/10" in description
        assert "155 minutes" in description
    
    def test_create_movie_descriptions(self, data_processor, sample_movie_data):
        """Test creation of descriptions DataFrame."""
        descriptions_df = data_processor.create_movie_descriptions(sample_movie_data)
        
        assert len(descriptions_df) == 3
        assert list(descriptions_df.columns) == ['Title', 'Description']
        assert all(title in descriptions_df['Title'].values for title in sample_movie_data['Title'])
    
    def test_chunk_descriptions(self, data_processor, sample_descriptions):
        """Test text chunking functionality."""
        chunks_df = data_processor.chunk_descriptions(sample_descriptions)
        
        assert len(chunks_df) >= 3  # At least one chunk per movie
        assert list(chunks_df.columns) == ['Title', 'Chunk', 'Metadata']
        
        # Check that metadata contains title
        for _, row in chunks_df.iterrows():
            assert 'Title' in row['Metadata']
            assert row['Metadata']['Title'] == row['Title']
    
    def test_load_movie_data(self, data_processor, sample_movie_data, temp_dir):
        """Test loading movie data from CSV."""
        file_path = os.path.join(temp_dir, "test_movies.csv")
        sample_movie_data.to_csv(file_path, index=False)
        
        loaded_data = data_processor.load_movie_data(file_path)
        
        assert len(loaded_data) == 3
        assert list(loaded_data.columns) == list(sample_movie_data.columns)
    
    def test_save_descriptions(self, data_processor, sample_descriptions, temp_dir):
        """Test saving descriptions to CSV."""
        file_path = os.path.join(temp_dir, "test_descriptions.csv")
        data_processor.save_descriptions(sample_descriptions, file_path)
        
        assert os.path.exists(file_path)
        
        # Load and verify
        loaded = pd.read_csv(file_path)
        assert len(loaded) == 3
        assert list(loaded.columns) == ['Title', 'Description']
    
    def test_save_chunks(self, data_processor, sample_descriptions, temp_dir):
        """Test saving chunks to CSV."""
        chunks_df = data_processor.chunk_descriptions(sample_descriptions)
        file_path = os.path.join(temp_dir, "test_chunks.csv")
        
        data_processor.save_chunks(chunks_df, file_path)
        
        assert os.path.exists(file_path)
        
        # Load and verify
        loaded = pd.read_csv(file_path)
        assert len(loaded) >= 3
        assert list(loaded.columns) == ['Title', 'Chunk', 'Metadata']