# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This is a Python-based movie recommendation system that uses Retrieval-Augmented Generation (RAG) with FAISS vector search and OpenAI's GPT models. The project follows standard Python packaging conventions with a modular architecture.

### Dependencies and Installation
- Install dependencies: `pip install -r requirements.txt`
- Install in development mode: `pip install -e .`
- Key libraries: langchain, faiss-cpu, openai, sentence-transformers, gradio, pandas

### Required Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI GPT models
- `OMDB_API_KEY`: Optional, for external movie data fetching

## Development Commands

### Running the Application
- **Web UI**: `python main.py --mode ui` or `python main.py`
- **CLI Mode**: `python main.py --mode cli`
- **Build vector store**: `python main.py --mode build`
- **Console script**: `movie-recommender --mode ui` (after `pip install -e .`)

### Data Setup and Processing
- Setup initial dataset: `python scripts/SetupData.py` (requires Kaggle API credentials)
- Format cast columns: `python scripts/format_movie.py`
- Clean composite dataset: `python scripts/FormatCastColumn.py`

### Testing
- Run all tests: `python run_tests.py` or `pytest tests/`
- Run specific test types: `python run_tests.py --type unit`
- Run with coverage: `python run_tests.py --coverage`
- Fast tests (skip slow): `python run_tests.py --fast`

### Code Quality
- Format code: `black src/ tests/`
- Lint code: `flake8 src/ tests/`
- Type checking: No specific command configured yet

## High-Level Architecture

### Project Structure
```
src/movie_recommender/     # Core application modules
├── config.py              # Configuration management
├── data_processor.py      # Data processing utilities  
├── vector_store.py        # FAISS vector operations
├── agents.py              # Multi-agent orchestration
├── rag_pipeline.py        # RAG implementation
├── ui.py                  # Gradio interface
└── main.py                # Application entry point

notebooks/                 # Jupyter development notebooks
scripts/                   # Data preparation scripts
tests/                     # Comprehensive test suite
```

### Core Components
1. **Data Pipeline**: `MovieDataProcessor` for dataset preparation and text chunking
2. **Vector Store**: `MovieVectorStore` with FAISS-based similarity search using sentence-transformers
3. **Agent Architecture**: Multi-agent system with retrieval, recommendation, and external data agents
4. **RAG Pipeline**: `MovieRAGPipeline` combining retrieval with OpenAI GPT for recommendations
5. **UI Layer**: Gradio interface with dual approaches (agents vs RAG pipeline)

### Key Files and Data Flow
- **Data Files**: 
  - `IMDb_Dataset_Composite_Cleaned.csv`: Main cleaned movie dataset
  - `Movie_Descriptions.csv`: Generated movie descriptions for embeddings
  - `movie_vector_store.index`: FAISS vector store
  - `movie_chunks_metadata.csv`: Metadata for vector chunks

- **Configuration**: `src/movie_recommender/config.py` - centralized settings
- **Entry Points**: `main.py` (standalone) or console script after installation

### Data Flow
1. Raw IMDb datasets → composite dataset via `scripts/SetupData.py`
2. Cast formatting via `scripts/format_movie.py` 
3. Movie descriptions generated from structured data via `MovieDataProcessor`
4. Text chunking and embedding creation using Sentence Transformers
5. FAISS index construction via `MovieVectorStore`
6. Two query paths:
   - **Agent Orchestrator**: Multi-agent workflow with structured recommendations
   - **RAG Pipeline**: Direct retrieval-augmented generation

### Agent Architecture
The system implements a multi-agent orchestrator pattern:
- **MovieRetrieverAgent**: FAISS vector store queries and result formatting
- **RecommendationAgent**: GPT-based recommendation generation with reasoning
- **ExternalDataAgent**: OMDB API integration for additional movie metadata
- **MovieRecommendationOrchestrator**: Coordinates agent interactions and error handling

### Testing Framework
- **Unit Tests**: Individual component testing (`test_config.py`, `test_data_processor.py`, etc.)
- **Integration Tests**: End-to-end workflow testing (`test_integration.py`)
- **Test Configuration**: `pytest.ini` with markers for slow/fast tests
- **Coverage Reporting**: HTML and terminal coverage reports available

### Running the System
1. **Development**: Execute cells in `notebooks/MovieFinder_Main.ipynb` for step-by-step exploration
2. **Production**: Use `python main.py --mode ui` for web interface or CLI mode for testing
3. **Installation**: Use `pip install -e .` for development installation with console scripts