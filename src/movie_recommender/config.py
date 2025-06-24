"""Configuration settings for the movie recommender system."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for the movie recommender system."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    omdb_api_key: Optional[str] = None
    
    # Model Settings
    embedding_provider: str = "openai"  # "openai" or "sentence_transformers"
    embedding_model_name: str = "text-embedding-3-small"  # OpenAI model or SentenceTransformer model
    llm_model_name: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    
    # Text Processing
    chunk_size: int = 200
    chunk_overlap: int = 20
    
    # Vector Store
    vector_store_path: str = "movie_vector_store.index"
    metadata_path: str = "movie_chunks_metadata.csv"
    
    # Data Files
    movie_data_path: str = "IMDb_Dataset_Composite_Cleaned.csv"
    movie_descriptions_path: str = "Movie_Descriptions.csv"
    
    # Search Settings
    default_top_k: int = 5
    
    # UI Settings
    gradio_server_name: str = "127.0.0.1"
    gradio_server_port: int = 7863
    
    def __post_init__(self):
        """Initialize API keys from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.omdb_api_key is None:
            self.omdb_api_key = os.getenv("OMDB_API_KEY")


def get_config() -> Config:
    """Get the default configuration."""
    return Config()