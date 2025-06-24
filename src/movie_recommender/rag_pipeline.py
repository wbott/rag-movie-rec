"""RAG (Retrieval-Augmented Generation) pipeline for movie recommendations."""

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import List, Dict

from .vector_store import MovieVectorStore


class MovieRAGPipeline:
    """RAG pipeline for movie recommendations using LangChain."""
    
    def __init__(self, vector_store: MovieVectorStore, llm: ChatOpenAI):
        self.vector_store = vector_store
        self.llm = llm
        self.rag_chain = self._create_rag_chain()
    
    def _create_rag_chain(self):
        """Create the RAG chain for movie recommendations."""
        prompt_template = """
        You are a movie expert. Based on the following movie descriptions, 
        answer the user's query as accurately and concisely as possible. 
        If the information is insufficient, say so.

        **Query**: {query}

        **Context**:
        {context}

        **Answer**:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the document processing chain
        rag_chain = (
            {
                "context": lambda x: self._format_docs(self.vector_store.search(x['query'])),
                "query": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _format_docs(self, results: List[Dict]) -> str:
        """Format search results for the prompt context."""
        return "\n\n".join([
            f"Title: {res['Title']}\nDescription: {res['Chunk']}"
            for res in results
        ])
    
    def query(self, query: str) -> str:
        """Process a query through the RAG pipeline."""
        try:
            response = self.rag_chain.invoke({"query": query})
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def batch_query(self, queries: List[str]) -> List[str]:
        """Process multiple queries through the RAG pipeline."""
        results = []
        for query in queries:
            results.append(self.query(query))
        return results