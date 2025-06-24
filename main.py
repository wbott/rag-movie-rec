#!/usr/bin/env python3
"""
Movie Recommender System - Main Entry Point

A chat-based movie search system using RAG (Retrieval-Augmented Generation)
with FAISS vector search and OpenAI GPT models.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from movie_recommender.main import main

if __name__ == "__main__":
    main()