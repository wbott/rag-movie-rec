"""Tests for agents module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from movie_recommender.agents import MovieRetrieverAgent, ExternalDataAgent, RecommendationAgent
from movie_recommender.vector_store import MovieVectorStore


class TestMovieRetrieverAgent:
    """Test cases for MovieRetrieverAgent."""
    
    def test_retrieve_movies_success(self):
        """Test successful movie retrieval."""
        # Mock vector store
        mock_vector_store = Mock(spec=MovieVectorStore)
        mock_vector_store.search.return_value = [
            {
                'Title': 'Test Movie',
                'Chunk': 'Test description',
                'Metadata': {'Title': 'Test Movie'},
                'Distance': 0.5
            }
        ]
        
        agent = MovieRetrieverAgent(mock_vector_store)
        result = agent.retrieve_movies.invoke({"query": "test query", "top_k": 1})
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['title'] == 'Test Movie'
        assert result[0]['description'] == 'Test description'
    
    def test_retrieve_movies_empty_query(self):
        """Test retrieval with empty query."""
        mock_vector_store = Mock(spec=MovieVectorStore)
        agent = MovieRetrieverAgent(mock_vector_store)
        
        result = agent.retrieve_movies.invoke({"query": "", "top_k": 1})
        
        assert isinstance(result, dict)
        assert "error" in result
    
    def test_retrieve_movies_no_results(self):
        """Test retrieval with no results."""
        mock_vector_store = Mock(spec=MovieVectorStore)
        mock_vector_store.search.return_value = []
        
        agent = MovieRetrieverAgent(mock_vector_store)
        result = agent.retrieve_movies.invoke({"query": "test query", "top_k": 1})
        
        assert isinstance(result, dict)
        assert "error" in result


class TestExternalDataAgent:
    """Test cases for ExternalDataAgent."""
    
    def test_external_agent_initialization(self):
        """Test external agent initialization."""
        agent = ExternalDataAgent(api_key="test-key")
        assert agent.api_key == "test-key"
    
    def test_fetch_movie_ratings_empty_title(self):
        """Test fetching ratings with empty title."""
        agent = ExternalDataAgent(api_key="test-key")
        result = agent.fetch_movie_ratings.invoke({"title": ""})
        
        assert isinstance(result, dict)
        assert "error" in result
    
    def test_fetch_movie_ratings_no_api_key(self):
        """Test fetching ratings without API key."""
        agent = ExternalDataAgent()
        result = agent.fetch_movie_ratings.invoke({"title": "Test Movie"})
        
        assert isinstance(result, dict)
        assert "error" in result
    
    @patch('requests.get')
    def test_fetch_movie_ratings_success(self, mock_get):
        """Test successful movie rating fetch."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "Response": "True",
            "Title": "Test Movie",
            "imdbRating": "8.0",
            "Year": "2020",
            "Metascore": "75",
            "Plot": "Test plot",
            "Actors": "Test Actor"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        agent = ExternalDataAgent(api_key="test-key")
        result = agent.fetch_movie_ratings.invoke({"title": "Test Movie"})
        
        assert isinstance(result, dict)
        assert "error" not in result
        assert result["title"] == "Test Movie"
        assert result["imdb_rating"] == "8.0"
    
    @patch('requests.get')
    def test_fetch_movie_ratings_not_found(self, mock_get):
        """Test movie not found response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "Response": "False",
            "Error": "Movie not found!"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        agent = ExternalDataAgent(api_key="test-key")
        result = agent.fetch_movie_ratings.invoke({"title": "Nonexistent Movie"})
        
        assert isinstance(result, dict)
        assert "error" in result


class TestRecommendationAgent:
    """Test cases for RecommendationAgent."""
    
    def test_recommendation_agent_initialization(self):
        """Test recommendation agent initialization."""
        mock_llm = Mock()
        mock_retriever = Mock()
        
        agent = RecommendationAgent(mock_llm, mock_retriever)
        
        assert agent.llm == mock_llm
        assert agent.retriever_agent == mock_retriever
    
    def test_recommend_movies_empty_query(self):
        """Test recommendation with empty query."""
        mock_llm = Mock()
        mock_retriever = Mock()
        
        agent = RecommendationAgent(mock_llm, mock_retriever)
        result = agent.recommend_movies.invoke({"query": "", "top_k": 3})
        
        assert isinstance(result, dict)
        assert "error" in result
    
    def test_recommend_movies_retrieval_error(self):
        """Test recommendation when retrieval fails."""
        mock_llm = Mock()
        mock_retriever = Mock()
        mock_retriever.retrieve_movies.invoke.return_value = {"error": "Retrieval failed"}
        
        agent = RecommendationAgent(mock_llm, mock_retriever)
        result = agent.recommend_movies.invoke({"query": "test query", "top_k": 3})
        
        assert isinstance(result, dict)
        assert "error" in result