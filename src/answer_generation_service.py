import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerationService:
    """
    Service for generating comprehensive answers from multimodal search results
    using Gemini 2.0 Flash via LangChain.
    """
    
    def __init__(self):
        """Initialize the answer generation service."""
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_tokens=8192, 
        )
        
        logger.info("Initialized AnswerGenerationService with Gemini 2.0 Flash")
    
    def prepare_bidirectional_context(self, search_results: Dict[str, List[Dict[str, Any]]], 
                                    n_articles: int = 3, n_images: int = 3) -> Dict[str, Any]:
        """
        Prepare bidirectional context: top N articles + their images AND top N images + their articles.
        
        Args:
            search_results: Dict with 'text' and 'images' keys containing search results
            n_articles: Number of top articles to include
            n_images: Number of top images to include
            
        Returns:
            Comprehensive context with both directions covered
        """
        context = {
            "primary_articles": [],
            "primary_images": [],
            "article_ids_covered": set(),
            "total_unique_sources": 0
        }
        
        top_articles = search_results.get('text', [])[:n_articles]
        for article in top_articles:
            article_id = article.get('article_id', '')
            context["article_ids_covered"].add(article_id)
            
            article_images = [
                img for img in search_results.get('images', []) 
                if img.get('article_id') == article_id
            ]
            
            context["primary_articles"].append({
                "title": article.get('title', ''),
                "content": article.get('content', ''),
                "article_id": article_id,
                "url": article.get('url', ''),
                "associated_images": article_images,
                "image_count": len(article_images)
            })
        
        # Get top N images and their source articles (if not already included)
        top_images = search_results.get('images', [])[:n_images]
        for image in top_images:
            article_id = image.get('article_id', '')
            
            # Find the source article for this image
            source_article = None
            if article_id not in context["article_ids_covered"]:
                # Look for the article in the search results or fetch it
                for article in search_results.get('text', []):
                    if article.get('article_id') == article_id:
                        source_article = article
                        break
                
                # If not found in current results, create placeholder with available info
                if not source_article:
                    source_article = {
                        "title": image.get('title', 'Article not in top text results'),
                        "content": f"Source article for image: {image.get('filename', '')}",
                        "article_id": article_id,
                        "url": image.get('url', ''),
                        "is_placeholder": True
                    }
                
                context["article_ids_covered"].add(article_id)
            
            context["primary_images"].append({
                "filename": image.get('filename', ''),
                "title": image.get('title', ''),
                "article_id": article_id,
                "source_article": source_article,
                "image_path": image.get('image_path', '')
            })
        
        context["total_unique_sources"] = len(context["article_ids_covered"])
        
        return context
    
    def format_bidirectional_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format the bidirectional context into a readable prompt format.
        
        Args:
            context: Prepared bidirectional context
            
        Returns:
            Formatted context string
        """
        formatted_context = []
        
        # Section 1: Primary Articles and their Images
        if context["primary_articles"]:
            formatted_context.append("PRIMARY ARTICLES WITH THEIR IMAGES:")
            for i, article in enumerate(context["primary_articles"], 1):
                formatted_context.append(f"\n{i}. Article: {article['title']}")
                formatted_context.append(f"   Content: {article['content']}")
                if article['url']:
                    formatted_context.append(f"   URL: {article['url']}")
                
                if article['associated_images']:
                    formatted_context.append(f"   Associated Images ({article['image_count']}):")
                    for img in article['associated_images']:
                        formatted_context.append(f"     - {img.get('filename', 'Unknown filename')}")
                else:
                    formatted_context.append("   No associated images found")
                formatted_context.append("")
        
        # Section 2: Primary Images and their Source Articles
        if context["primary_images"]:
            formatted_context.append("\nPRIMARY IMAGES WITH THEIR SOURCE ARTICLES:")
            for i, image in enumerate(context["primary_images"], 1):
                formatted_context.append(f"\n{i}. Image: {image['filename']}")
                formatted_context.append(f"   From Article: {image['title']}")
                
                if image['source_article']:
                    source = image['source_article']
                    if not source.get('is_placeholder', False):
                        formatted_context.append(f"   Article Content: {source.get('content', '')}")
                    else:
                        formatted_context.append(f"   Note: This image's source article was not in the top text results")
                    
                    if source.get('url'):
                        formatted_context.append(f"   Article URL: {source['url']}")
                formatted_context.append("")
        
        # Summary
        formatted_context.append(f"\nSUMMARY: Drawing from {context['total_unique_sources']} unique sources with both textual and visual content.")
        
        return "\n".join(formatted_context)
    
    def create_system_prompt(self) -> str:
        """Create system prompt for answer generation."""
        return """You are an expert AI assistant helping users understand information from news articles and research papers. Your task is to provide comprehensive, accurate answers based on the provided sources.

Guidelines:
- Synthesize information from multiple sources when relevant
- Provide clear, well-structured answers
- Focus on the most important and relevant information
- When visual content is available, acknowledge its relevance to the topic
- Be concise but thorough
- If sources don't fully answer the question, acknowledge the limitations"""
    
    def generate_answer(self, query: str, search_results: Dict[str, List[Dict[str, Any]]], 
                       n_articles: int = 3, n_images: int = 3) -> str:
        """
        Generate a comprehensive answer using bidirectional multimodal context.
        
        Args:
            query: User's question
            search_results: Multimodal search results
            n_articles: Number of top articles to include with their images
            n_images: Number of top images to include with their articles
            
        Returns:
            Generated answer
        """
        try:
            # Prepare bidirectional context
            context = self.prepare_bidirectional_context(search_results, n_articles, n_images)
            
            if context["total_unique_sources"] == 0:
                return "I couldn't find relevant information to answer your question. Please try rephrasing your query or asking about a different topic."
            
            # Format context for prompt
            formatted_context = self.format_bidirectional_context_for_prompt(context)
            
            # Create the prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.create_system_prompt()),
                ("human", """Question: {question}

Available Information (Bidirectional Multimodal Context):
{context}

Please provide a comprehensive answer based on the available information. Consider both the textual content and the visual elements mentioned. When relevant, explain how the visual content supports or illustrates the textual information. If question oriented on finding specific images, describe them based on the images provided.""")
            ])
            
            # Format the prompt
            messages = prompt_template.format_messages(
                question=query,
                context=formatted_context
            )
            
            # Generate answer
            response = self.llm.invoke(messages)
            
            logger.info(f"Generated answer for query: {query[:50]}... using {context['total_unique_sources']} sources")
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while processing your question. Please try again."
    
    def generate_answer_with_summary(self, query: str, search_results: Dict[str, List[Dict[str, Any]]], 
                                   n_articles: int = 3, n_images: int = 3) -> Dict[str, Any]:
        """
        Generate answer with additional summary information using bidirectional context.
        
        Args:
            query: User's question
            search_results: Multimodal search results
            n_articles: Number of top articles to include
            n_images: Number of top images to include
            
        Returns:
            Dict containing answer and metadata
        """
        context = self.prepare_bidirectional_context(search_results, n_articles, n_images)
        answer = self.generate_answer(query, search_results, n_articles, n_images)
        
        return {
            "answer": answer,
            "sources_used": context["total_unique_sources"],
            "primary_articles": len(context["primary_articles"]),
            "primary_images": len(context["primary_images"]),
            "has_bidirectional_context": len(context["primary_articles"]) > 0 and len(context["primary_images"]) > 0,
            "context_type": "bidirectional_multimodal"
        }
    
    def enhance_search_results_with_images(self, search_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enhance text search results by finding related images.
        This integrates textual and visual data for better context.
        
        Args:
            search_results: Current search results
            
        Returns:
            Enhanced search results with related images integrated
        """
        enhanced_results = search_results.copy()
        
        # Get article IDs from text results
        text_article_ids = {result.get('article_id') for result in search_results.get('text', [])}
        
        # Add images from the same articles to provide visual context
        if 'images' in search_results:
            related_images = []
            standalone_images = []
            
            for image in search_results['images']:
                if image.get('article_id') in text_article_ids:
                    related_images.append(image)
                else:
                    standalone_images.append(image)
            
            # Prioritize related images, then add standalone ones
            enhanced_results['images'] = related_images + standalone_images[:3]  # Limit total images
        
        # Add image information to text results
        for text_result in enhanced_results.get('text', []):
            article_id = text_result.get('article_id')
            related_imgs = [img for img in enhanced_results.get('images', []) 
                          if img.get('article_id') == article_id]
            text_result['associated_images'] = related_imgs
            text_result['has_visual_content'] = len(related_imgs) > 0
        
        return enhanced_results


def main():
    """Test the answer generation service with bidirectional context."""
    try:
        service = AnswerGenerationService()
        
        # Mock search results for testing bidirectional approach
        mock_results = {
            "text": [
                {
                    "title": "AI Advances in Healthcare",
                    "content": "Recent developments in artificial intelligence have shown promising results in medical diagnosis and treatment planning. Machine learning algorithms can now detect diseases earlier and with higher accuracy than traditional methods.",
                    "article_id": "1",
                    "url": "https://example.com/ai-healthcare"
                },
                {
                    "title": "Autonomous Vehicles Progress",
                    "content": "Self-driving car technology has made significant strides with improved sensor technology and neural networks for decision making.",
                    "article_id": "2", 
                    "url": "https://example.com/autonomous-vehicles"
                }
            ],
            "images": [
                {
                    "title": "AI Advances in Healthcare", 
                    "filename": "1_ai_healthcare_chart.png",
                    "article_id": "1",
                    "image_path": "data/img/1_ai_healthcare_chart.png"
                },
                {
                    "title": "Robotics in Manufacturing",
                    "filename": "3_robotics_factory.jpg", 
                    "article_id": "3",
                    "image_path": "data/img/3_robotics_factory.jpg"
                }
            ]
        }
        
        test_query = "How is AI being used in different industries?"
        
        # Generate answer with bidirectional context
        result = service.generate_answer_with_summary(test_query, mock_results, n_articles=2, n_images=2)
        
        print(f"Question: {test_query}")
        print(f"Answer: {result['answer']}")
        print(f"Sources used: {result['sources_used']}")
        print(f"Primary articles: {result['primary_articles']}")
        print(f"Primary images: {result['primary_images']}")
        print(f"Has bidirectional context: {result['has_bidirectional_context']}")
        
    except Exception as e:
        print(f"Error testing service: {e}")


if __name__ == "__main__":
    main()