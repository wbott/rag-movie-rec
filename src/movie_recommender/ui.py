"""Gradio UI for the movie recommendation system."""

import gradio as gr
from typing import Dict

from .agents import MovieRecommendationOrchestrator
from .rag_pipeline import MovieRAGPipeline


class MovieRecommenderUI:
    """Gradio-based user interface for movie recommendations."""
    
    def __init__(self, orchestrator: MovieRecommendationOrchestrator, rag_pipeline: MovieRAGPipeline):
        self.orchestrator = orchestrator
        self.rag_pipeline = rag_pipeline
    
    def process_query_orchestrator(self, query: str) -> str:
        """Process query using the orchestrator approach."""
        if not query.strip():
            return "Please enter a movie query."
        
        result = self.orchestrator.process_query(query)
        
        if result.get("error"):
            return f"Error: {result['error']}"
        
        output = "**🎬 Movie Recommendations**:\n\n"
        for rec in result.get("recommendations", []):
            output += f"• **{rec['title']}**: {rec['reason']}\n\n"
        
        if result.get("retrieved_movies"):
            output += "\n**📚 Retrieved Context**:\n\n"
            for movie in result["retrieved_movies"][:3]:
                output += f"• {movie['title']}: {movie['description'][:100]}...\n\n"
        
        return output
    
    def process_query_rag(self, query: str) -> str:
        """Process query using the RAG pipeline approach."""
        if not query.strip():
            return "Please enter a movie query."
        
        return self.rag_pipeline.query(query)
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="🎬 Movie Recommender", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🎬 Movie Recommendation System
            
            **Ask about movies by genre, actors, plot summaries, or reviews — just like chatting with a friend.**
            
            Choose between two approaches:
            - **Orchestrator**: Uses multiple agents for structured recommendations
            - **RAG Pipeline**: Direct retrieval-augmented generation
            """)
            
            with gr.Tabs():
                with gr.TabItem("🤖 Agent Orchestrator"):
                    with gr.Row():
                        with gr.Column():
                            query_input_1 = gr.Textbox(
                                lines=3,
                                placeholder="e.g., 'Recommend a documentary about a famous person'",
                                label="Your Movie Query"
                            )
                            submit_btn_1 = gr.Button("Get Recommendations", variant="primary")
                        
                        with gr.Column():
                            output_1 = gr.Textbox(
                                lines=15,
                                label="Recommendations",
                                interactive=False
                            )
                    
                    submit_btn_1.click(
                        fn=self.process_query_orchestrator,
                        inputs=[query_input_1],
                        outputs=output_1
                    )
                
                with gr.TabItem("🔗 RAG Pipeline"):
                    with gr.Row():
                        with gr.Column():
                            query_input_2 = gr.Textbox(
                                lines=3,
                                placeholder="e.g., 'What are some good action movies with high ratings?'",
                                label="Your Movie Query"
                            )
                            submit_btn_2 = gr.Button("Ask", variant="primary")
                        
                        with gr.Column():
                            output_2 = gr.Textbox(
                                lines=15,
                                label="Response",
                                interactive=False
                            )
                    
                    submit_btn_2.click(
                        fn=self.process_query_rag,
                        inputs=[query_input_2],
                        outputs=output_2
                    )
            
            gr.Markdown("""
            ---
            **💡 Example Queries:**
            - "Recommend documentaries about famous people"
            - "What are some good action movies from the 2000s?"
            - "Movies similar to The Lord of the Rings"
            - "Comedy movies with high IMDb ratings"
            """)
        
        return demo
    
    def launch(self, server_name: str = "127.0.0.1", server_port: int = 7863, share: bool = False):
        """Launch the Gradio interface."""
        demo = self.create_interface()
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        )