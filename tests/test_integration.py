"""Integration tests for the movie recommender system."""

import pytest
import os
import tempfile
import pandas as pd
from unittest.mock import Mock, patch

from movie_recommender.main import MovieRecommenderApp
from movie_recommender.data_processor import MovieDataProcessor
from movie_recommender.vector_store import MovieVectorStore


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def sample_system_data(self, temp_dir):
        """Create sample data files for integration testing."""
        # Create sample movie data
        movie_data = pd.DataFrame({
            'Title': ['Gladiator', 'Inception', 'The Matrix', 'Pulp Fiction', 'The Godfather'],
            'Year': [2000, 2010, 1999, 1994, 1972],
            'Genres': ['Action, Drama', 'Sci-Fi, Thriller', 'Action, Sci-Fi', 'Crime, Drama', 'Crime, Drama'],
            'Director': ['Ridley Scott', 'Christopher Nolan', 'The Wachowskis', 'Quentin Tarantino', 'Francis Ford Coppola'],
            'Star Cast': ['Russell Crowe, Joaquin Phoenix', 'Leonardo DiCaprio, Marion Cotillard', 'Keanu Reeves, Laurence Fishburne', 'John Travolta, Samuel L. Jackson', 'Marlon Brando, Al Pacino'],
            'IMDb Rating': [8.5, 8.8, 8.7, 8.9, 9.2],
            'Duration (minutes)': [155, 148, 136, 154, 175],
            'Certificates': ['R', 'PG-13', 'R', 'R', 'R'],
            'MetaScore': [67.0, 74.0, 73.0, 95.0, 100.0]
        })
        
        # Save to temp directory
        movie_path = os.path.join(temp_dir, "IMDb_Dataset_Composite_Cleaned.csv")
        movie_data.to_csv(movie_path, index=False)
        
        return {
            'movie_path': movie_path,
            'temp_dir': temp_dir,
            'movie_data': movie_data
        }
    
    def test_data_processing_pipeline(self, sample_system_data):
        """Test the complete data processing pipeline."""
        data_processor = MovieDataProcessor(chunk_size=100, chunk_overlap=10)
        
        # Load data
        movie_df = data_processor.load_movie_data(sample_system_data['movie_path'])
        assert len(movie_df) == 5
        
        # Create descriptions
        descriptions_df = data_processor.create_movie_descriptions(movie_df)
        assert len(descriptions_df) == 5
        assert all('Description' in desc for desc in descriptions_df['Description'])
        
        # Create chunks
        chunks_df = data_processor.chunk_descriptions(descriptions_df)
        assert len(chunks_df) >= 5  # Should have at least one chunk per movie
        
        # Verify chunk structure
        assert list(chunks_df.columns) == ['Title', 'Chunk', 'Metadata']
        for _, row in chunks_df.iterrows():
            assert isinstance(row['Metadata'], dict)
            assert 'Title' in row['Metadata']
    
    def test_vector_store_pipeline(self, sample_system_data):
        """Test the vector store creation and search pipeline."""
        # Process data
        data_processor = MovieDataProcessor(chunk_size=100, chunk_overlap=10)
        movie_df = data_processor.load_movie_data(sample_system_data['movie_path'])
        descriptions_df = data_processor.create_movie_descriptions(movie_df)
        chunks_df = data_processor.chunk_descriptions(descriptions_df)
        
        # Build vector store
        vector_store = MovieVectorStore(embedding_model_name="all-MiniLM-L6-v2")
        vector_store.build_index(chunks_df)
        
        # Test search functionality
        results = vector_store.search("action movie", k=3)
        assert len(results) == 3
        assert all('Title' in result for result in results)
        
        # Test save and load
        index_path = os.path.join(sample_system_data['temp_dir'], "test_index.index")
        metadata_path = os.path.join(sample_system_data['temp_dir'], "test_metadata.csv")
        
        vector_store.save_index(index_path, metadata_path)
        assert os.path.exists(index_path)
        assert os.path.exists(metadata_path)
        
        # Load in new instance
        new_store = MovieVectorStore(embedding_model_name="all-MiniLM-L6-v2")
        new_store.load_index(index_path, metadata_path)
        
        # Test search in loaded store
        new_results = new_store.search("sci-fi movie", k=2)
        assert len(new_results) == 2
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_app_initialization(self, sample_system_data):
        """Test MovieRecommenderApp initialization."""
        # Change working directory context for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(sample_system_data['temp_dir'])
            
            # Copy the movie data to expected location
            movie_data_path = "IMDb_Dataset_Composite_Cleaned.csv"
            if not os.path.exists(movie_data_path):
                sample_system_data['movie_data'].to_csv(movie_data_path, index=False)
            
            app = MovieRecommenderApp()
            
            # Test data pipeline setup
            app.setup_data_pipeline()
            assert app.data_processor is not None
            
            # Test vector store setup
            app.setup_vector_store()
            assert app.vector_store is not None
            
            # Test stats before building
            stats = app.vector_store.get_stats()
            assert stats["status"] == "not_loaded"
            
        finally:
            os.chdir(original_cwd)
    
    def test_search_quality(self, sample_system_data):
        """Test that search returns relevant results."""
        # Build complete pipeline
        data_processor = MovieDataProcessor(chunk_size=100, chunk_overlap=10)
        movie_df = data_processor.load_movie_data(sample_system_data['movie_path'])
        descriptions_df = data_processor.create_movie_descriptions(movie_df)
        chunks_df = data_processor.chunk_descriptions(descriptions_df)
        
        vector_store = MovieVectorStore(embedding_model_name="all-MiniLM-L6-v2")
        vector_store.build_index(chunks_df)
        
        # Test specific queries
        action_results = vector_store.search("action movie with gladiators", k=3)
        sci_fi_results = vector_store.search("science fiction dream movie", k=3)
        crime_results = vector_store.search("crime drama classic", k=3)
        
        # Verify we get results
        assert len(action_results) == 3
        assert len(sci_fi_results) == 3
        assert len(crime_results) == 3
        
        # Basic relevance check - action query should include action movies
        action_titles = [result['Title'] for result in action_results]
        assert any(title in ['Gladiator', 'The Matrix'] for title in action_titles)
        
        # Sci-fi query should prefer sci-fi movies
        sci_fi_titles = [result['Title'] for result in sci_fi_results]
        assert any(title in ['Inception', 'The Matrix'] for title in sci_fi_titles)
    
    def test_end_to_end_workflow(self, sample_system_data):
        """Test a complete end-to-end workflow."""
        # This test simulates the full user workflow
        
        # 1. Data preparation
        data_processor = MovieDataProcessor()
        movie_df = pd.read_csv(sample_system_data['movie_path'])
        
        # 2. Description generation
        descriptions_df = data_processor.create_movie_descriptions(movie_df)
        
        # 3. Text chunking
        chunks_df = data_processor.chunk_descriptions(descriptions_df)
        
        # 4. Vector store creation
        vector_store = MovieVectorStore()
        vector_store.build_index(chunks_df)
        
        # 5. Save vector store
        index_path = os.path.join(sample_system_data['temp_dir'], "workflow_index.index")
        metadata_path = os.path.join(sample_system_data['temp_dir'], "workflow_metadata.csv")
        vector_store.save_index(index_path, metadata_path)
        
        # 6. Load in fresh instance (simulating app restart)
        fresh_store = MovieVectorStore()
        fresh_store.load_index(index_path, metadata_path)
        
        # 7. Perform searches
        queries = [
            "action movies",
            "science fiction",
            "crime drama",
            "movies with good ratings"
        ]
        
        for query in queries:
            results = fresh_store.search(query, k=2)
            assert len(results) == 2
            assert all('Title' in result for result in results)
            assert all('Chunk' in result for result in results)
        
        # 8. Verify system statistics
        stats = fresh_store.get_stats()
        assert stats["status"] == "loaded"
        assert stats["unique_movies"] == 5
        assert stats["total_vectors"] > 0