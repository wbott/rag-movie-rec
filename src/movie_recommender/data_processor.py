"""Data processing utilities for movie descriptions and embeddings."""

import pandas as pd
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MovieDataProcessor:
    """Handles movie data processing and description generation."""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def generate_description(self, row: pd.Series) -> str:
        """Generate a movie description from DataFrame row."""
        title = row['Title']
        year = int(row['Year'])
        genres = row['Genres']
        director = row['Director']
        star_cast = self.format_star_cast(row['Star Cast'])
        rating = row['IMDb Rating']
        duration = int(row['Duration (minutes)'])
        certificate = row['Certificates']
        metascore = row['MetaScore']
        
        description = (
            f"{title} ({year}) is a {genres.lower()} film directed by {director}. "
            f"Featuring {star_cast}, this movie has an IMDb rating of {rating}/10 and a MetaScore of {metascore}. "
            f"With a runtime of {duration} minutes, it is rated {certificate}."
        )
        return description
    
    def format_star_cast(self, star_cast: str) -> str:
        """Format star cast string for better readability."""
        actors = [actor.strip() for actor in star_cast.split(',')]
        if len(actors) > 3:
            return ", ".join(actors[:3]) + ", and others"
        return ", ".join(actors)
    
    def create_movie_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create descriptions for all movies in the DataFrame."""
        descriptions = []
        for _, row in df.iterrows():
            desc = self.generate_description(row)
            descriptions.append({'Title': row['Title'], 'Description': desc})
        
        return pd.DataFrame(descriptions)
    
    def chunk_descriptions(self, descriptions_df: pd.DataFrame) -> pd.DataFrame:
        """Split descriptions into chunks with metadata."""
        chunks = []
        for _, row in descriptions_df.iterrows():
            split_texts = self.text_splitter.split_text(row['Description'])
            for chunk in split_texts:
                chunks.append({
                    'Title': row['Title'],
                    'Chunk': chunk,
                    'Metadata': {'Title': row['Title']}
                })
        
        return pd.DataFrame(chunks)
    
    def load_movie_data(self, file_path: str) -> pd.DataFrame:
        """Load movie data from CSV file."""
        return pd.read_csv(file_path)
    
    def save_descriptions(self, descriptions_df: pd.DataFrame, file_path: str) -> None:
        """Save descriptions to CSV file."""
        descriptions_df.to_csv(file_path, index=False)
    
    def save_chunks(self, chunks_df: pd.DataFrame, file_path: str) -> None:
        """Save chunks to CSV file."""
        chunks_df[['Title', 'Chunk', 'Metadata']].to_csv(file_path, index=False)