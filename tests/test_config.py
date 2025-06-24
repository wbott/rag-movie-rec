"""Tests for configuration module."""

import pytest
import os
from movie_recommender.config import Config, get_config


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_default_values(self):
        """Test that config has sensible default values."""
        config = Config()
        
        assert config.embedding_model_name == "all-MiniLM-L6-v2"
        assert config.llm_model_name == "gpt-3.5-turbo"
        assert config.llm_temperature == 0.7
        assert config.llm_max_tokens == 512
        assert config.chunk_size == 200
        assert config.chunk_overlap == 20
        assert config.default_top_k == 5
    
    def test_config_with_custom_values(self):
        """Test config with custom values."""
        config = Config(
            embedding_model_name="custom-model",
            llm_temperature=0.5,
            chunk_size=150
        )
        
        assert config.embedding_model_name == "custom-model"
        assert config.llm_temperature == 0.5
        assert config.chunk_size == 150
    
    def test_config_env_variables(self, monkeypatch):
        """Test that config picks up environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("OMDB_API_KEY", "test-omdb-key")
        
        config = Config()
        
        assert config.openai_api_key == "test-openai-key"
        assert config.omdb_api_key == "test-omdb-key"
    
    def test_get_config_function(self):
        """Test the get_config function."""
        config = get_config()
        
        assert isinstance(config, Config)
        assert config.embedding_model_name == "all-MiniLM-L6-v2"