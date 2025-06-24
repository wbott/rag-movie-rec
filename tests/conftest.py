"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from movie_recommender.config import Config
from movie_recommender.data_processor import MovieDataProcessor
from movie_recommender.vector_store import MovieVectorStore


@pytest.fixture
def sample_movie_data():
    """Sample movie data for testing."""
    return pd.DataFrame({
        'Title': ['Gladiator', 'Inception', 'The Matrix'],
        'Year': [2000, 2010, 1999],
        'Genres': ['Action, Drama', 'Sci-Fi, Thriller', 'Action, Sci-Fi'],
        'Director': ['Ridley Scott', 'Christopher Nolan', 'The Wachowskis'],
        'Star Cast': ['Russell Crowe, Joaquin Phoenix', 'Leonardo DiCaprio, Marion Cotillard', 'Keanu Reeves, Laurence Fishburne'],
        'IMDb Rating': [8.5, 8.8, 8.7],
        'Duration (minutes)': [155, 148, 136],
        'Certificates': ['R', 'PG-13', 'R'],
        'MetaScore': [67.0, 74.0, 73.0]
    })


@pytest.fixture
def sample_descriptions():
    """Sample movie descriptions for testing."""
    return pd.DataFrame({
        'Title': ['Gladiator', 'Inception', 'The Matrix'],
        'Description': [
            'Gladiator (2000) is a action, drama film directed by Ridley Scott.',
            'Inception (2010) is a sci-fi, thriller film directed by Christopher Nolan.',
            'The Matrix (1999) is a action, sci-fi film directed by The Wachowskis.'
        ]
    })


@pytest.fixture
def test_config():
    """Test configuration."""
    return Config(
        openai_api_key="test-key",
        chunk_size=100,
        chunk_overlap=10,
        default_top_k=3
    )


@pytest.fixture
def data_processor():
    """Data processor instance for testing."""
    return MovieDataProcessor(chunk_size=100, chunk_overlap=10)


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    # Use a small embedding model for faster tests
    return MovieVectorStore(embedding_model_name="all-MiniLM-L6-v2")