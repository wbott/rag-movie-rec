"""Main application entry point for the movie recommender system."""

import os
import sys
from pathlib import Path
from langchain_openai import ChatOpenAI

from .config import get_config
from .data_processor import MovieDataProcessor
from .vector_store import MovieVectorStore
from .agents import MovieRecommendationOrchestrator
from .rag_pipeline import MovieRAGPipeline
from .ui import MovieRecommenderUI


class MovieRecommenderApp:
    """Main application class for the movie recommender system."""
    
    def __init__(self, config_path: str = None):
        self.config = get_config()
        self.data_processor = None
        self.vector_store = None
        self.llm = None
        self.orchestrator = None
        self.rag_pipeline = None
        self.ui = None
        
        # Validate API keys
        if not self.config.openai_api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
    
    def setup_data_pipeline(self):
        """Set up the data processing pipeline."""
        print("🔧 Setting up data pipeline...")
        self.data_processor = MovieDataProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        print("✅ Data processor initialized")
    
    def setup_vector_store(self):
        """Set up the vector store."""
        print("🔧 Setting up vector store...")
        self.vector_store = MovieVectorStore(
            embedding_model_name=self.config.embedding_model_name
        )
        print("✅ Vector store initialized")
    
    def setup_llm(self):
        """Set up the language model."""
        if not self.config.openai_api_key:
            api_key = input("Please enter your OpenAI API key: ")
            if not api_key:
                raise ValueError("OpenAI API key is required")
            self.config.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        
        print("🔧 Setting up language model...")
        self.llm = ChatOpenAI(
            model_name=self.config.llm_model_name,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            openai_api_key=self.config.openai_api_key
        )
        print("✅ Language model initialized")
    
    def process_data(self):
        """Process movie data and create descriptions."""
        if not self.data_processor:
            self.setup_data_pipeline()
        
        print("📊 Processing movie data...")
        
        # Load movie data
        if not os.path.exists(self.config.movie_data_path):
            raise FileNotFoundError(f"Movie data file not found: {self.config.movie_data_path}")
        
        movie_df = self.data_processor.load_movie_data(self.config.movie_data_path)
        print(f"✅ Loaded {len(movie_df)} movies")
        
        # Create descriptions
        descriptions_df = self.data_processor.create_movie_descriptions(movie_df)
        self.data_processor.save_descriptions(descriptions_df, self.config.movie_descriptions_path)
        print(f"✅ Created descriptions for {len(descriptions_df)} movies")
        
        # Create chunks
        chunks_df = self.data_processor.chunk_descriptions(descriptions_df)
        print(f"✅ Created {len(chunks_df)} text chunks")
        
        return chunks_df
    
    def build_vector_store(self, chunks_df=None):
        """Build the vector store from movie chunks."""
        if not self.vector_store:
            self.setup_vector_store()
        
        if chunks_df is None:
            chunks_df = self.process_data()
        
        print("🧬 Building vector store...")
        self.vector_store.build_index(chunks_df)
        
        print("💾 Saving vector store...")
        self.vector_store.save_index(self.config.vector_store_path, self.config.metadata_path)
        
        print("✅ Vector store built and saved")
    
    def load_vector_store(self):
        """Load existing vector store."""
        if not self.vector_store:
            self.setup_vector_store()
        
        if not os.path.exists(self.config.vector_store_path):
            print("❌ Vector store not found. Building new one...")
            self.build_vector_store()
            return
        
        print("📂 Loading vector store...")
        self.vector_store.load_index(self.config.vector_store_path, self.config.metadata_path)
        
        stats = self.vector_store.get_stats()
        print(f"✅ Vector store loaded: {stats['total_vectors']} vectors, {stats['unique_movies']} movies")
    
    def setup_agents(self):
        """Set up the agent orchestrator."""
        if not self.llm:
            self.setup_llm()
        if not self.vector_store:
            self.load_vector_store()
        
        print("🤖 Setting up agents...")
        self.orchestrator = MovieRecommendationOrchestrator(
            vector_store=self.vector_store,
            llm=self.llm,
            omdb_api_key=self.config.omdb_api_key
        )
        print("✅ Agent orchestrator initialized")
    
    def setup_rag_pipeline(self):
        """Set up the RAG pipeline."""
        if not self.llm:
            self.setup_llm()
        if not self.vector_store:
            self.load_vector_store()
        
        print("🔗 Setting up RAG pipeline...")
        self.rag_pipeline = MovieRAGPipeline(
            vector_store=self.vector_store,
            llm=self.llm
        )
        print("✅ RAG pipeline initialized")
    
    def setup_ui(self):
        """Set up the user interface."""
        if not self.orchestrator:
            self.setup_agents()
        if not self.rag_pipeline:
            self.setup_rag_pipeline()
        
        print("🎨 Setting up user interface...")
        self.ui = MovieRecommenderUI(
            orchestrator=self.orchestrator,
            rag_pipeline=self.rag_pipeline
        )
        print("✅ User interface initialized")
    
    def run_cli_demo(self):
        """Run a command-line demo."""
        if not self.orchestrator:
            self.setup_agents()
        
        print("\n🎬 Movie Recommender CLI Demo")
        print("=" * 40)
        
        while True:
            query = input("\nEnter your movie query (or 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("\n🔍 Processing query...")
            result = self.orchestrator.process_query(query)
            
            if result.get("error"):
                print(f"❌ Error: {result['error']}")
            else:
                print("\n🎯 Recommendations:")
                for i, rec in enumerate(result.get("recommendations", []), 1):
                    print(f"{i}. {rec['title']}: {rec['reason']}")
    
    def launch_ui(self, share: bool = False):
        """Launch the web UI."""
        if not self.ui:
            self.setup_ui()
        
        print(f"\n🚀 Launching web interface...")
        print(f"📍 Access at: http://{self.config.gradio_server_name}:{self.config.gradio_server_port}")
        
        self.ui.launch(
            server_name=self.config.gradio_server_name,
            server_port=self.config.gradio_server_port,
            share=share
        )


def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Movie Recommender System")
    parser.add_argument("--mode", choices=["build", "cli", "ui"], default="ui",
                       help="Mode to run the application in")
    parser.add_argument("--share", action="store_true",
                       help="Share the UI publicly (Gradio)")
    
    args = parser.parse_args()
    
    app = MovieRecommenderApp()
    
    try:
        if args.mode == "build":
            print("🏗️  Building movie recommender system...")
            app.build_vector_store()
            print("✅ Build complete!")
        
        elif args.mode == "cli":
            print("🖥️  Starting CLI demo...")
            app.run_cli_demo()
        
        elif args.mode == "ui":
            print("🌐 Starting web interface...")
            app.launch_ui(share=args.share)
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()