"""Agent implementations for movie recommendation system."""

import requests
import os
from typing import List, Dict, Any
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .vector_store import MovieVectorStore


class MovieRetrieverAgent:
    """Agent for retrieving movies from FAISS vector store."""
    
    def __init__(self, vector_store: MovieVectorStore):
        self.vector_store = vector_store
    
    @tool
    def retrieve_movies(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve movies from the FAISS vector store based on a query."""
        if not query.strip():
            return {"error": "Query cannot be empty"}
        
        try:
            results = self.vector_store.search(query, k=top_k)
            if not results:
                return {"error": "No movies found"}
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result['Metadata'].get("Title"),
                    "description": result['Chunk'],
                    "metadata": result['Metadata'],
                    "distance": result.get('Distance', 0.0)
                })
            return formatted_results
        except Exception as e:
            return {"error": f"Retrieval error: {str(e)}"}


class ExternalDataAgent:
    """Agent for fetching external movie data from OMDB API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OMDB_API_KEY")
    
    @tool
    def fetch_movie_ratings(self, title: str) -> Dict:
        """Fetch movie ratings from the OMDB API."""
        if not title.strip():
            return {"error": "Movie title cannot be empty"}
        
        if not self.api_key:
            return {"error": "OMDB API key is required"}
        
        url = "http://www.omdbapi.com/"
        params = {
            "t": title,
            "apikey": self.api_key,
            "plot": "full"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("Response") == "True":
                return {
                    "title": data.get("Title", title),
                    "imdb_rating": data.get("imdbRating", "N/A"),
                    "source": "OMDB API",
                    "year": data.get("Year", "N/A"),
                    "metascore": data.get("Metascore", "N/A"),
                    "plot": data.get("Plot", "N/A"),
                    "actors": data.get("Actors", "N/A")
                }
            else:
                return {"error": f"Movie not found: {data.get('Error', 'Unknown error')}"}
        
        except requests.RequestException as e:
            return {"error": f"OMDB API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


class RecommendationAgent:
    """Agent for generating movie recommendations."""
    
    def __init__(self, llm: ChatOpenAI, retriever_agent: MovieRetrieverAgent):
        self.llm = llm
        self.retriever_agent = retriever_agent
    
    @tool
    def recommend_movies(self, query: str, top_k: int = 3) -> List[Dict]:
        """Generate movie recommendations based on the query."""
        if not query.strip():
            return {"error": "Query cannot be empty"}
        
        try:
            # Retrieve movies
            retrieved = self.retriever_agent.retrieve_movies.invoke({"query": query, "top_k": 5})
            if isinstance(retrieved, dict) and "error" in retrieved:
                return retrieved
            
            # Generate recommendations
            prompt_template = """
            You are a movie expert. Based on the user query: '{query}',
            recommend {top_k} movies from the following list:
            {movies}
            
            Provide a brief reason for each recommendation in the format:
            - Title: Reason
            
            Response:
            """
            prompt = PromptTemplate.from_template(prompt_template)
            movies_str = "\n".join([f"{m['title']}: {m['description']}" for m in retrieved])
            chain = prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "query": query,
                "top_k": top_k,
                "movies": movies_str
            })
            
            # Parse response
            recommendations = []
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("-") and ":" in line:
                    line = line[1:].strip()  # Remove the dash
                    if ":" in line:
                        title, reason = line.split(":", 1)
                        recommendations.append({
                            "title": title.strip(),
                            "reason": reason.strip()
                        })
            
            return recommendations if recommendations else {"error": "No recommendations generated"}
        
        except Exception as e:
            return {"error": f"Recommendation error: {str(e)}"}


class MovieRecommendationOrchestrator:
    """Orchestrates the movie recommendation process using multiple agents."""
    
    def __init__(self, vector_store: MovieVectorStore, llm: ChatOpenAI, omdb_api_key: str = None):
        self.retriever = MovieRetrieverAgent(vector_store)
        self.external_data = ExternalDataAgent(omdb_api_key)
        self.recommender = RecommendationAgent(llm, self.retriever)
    
    def process_query(self, query: str, top_k: int = 3) -> Dict:
        """Orchestrate the recommendation process."""
        try:
            # Retrieve movies
            retrieved = self.retriever.retrieve_movies.invoke({"query": query, "top_k": 5})
            if isinstance(retrieved, dict) and "error" in retrieved:
                return {"error": retrieved["error"], "recommendations": [], "ratings": []}
            
            # Generate recommendations
            recommendations = self.recommender.recommend_movies.invoke({
                "query": query,
                "top_k": top_k
            })
            
            if isinstance(recommendations, dict) and "error" in recommendations:
                return {"error": recommendations["error"], "recommendations": [], "ratings": []}
            
            return {
                "recommendations": recommendations,
                "retrieved_movies": retrieved[:top_k],
                "error": None
            }
        
        except Exception as e:
            return {"error": f"Orchestration error: {str(e)}", "recommendations": [], "ratings": []}