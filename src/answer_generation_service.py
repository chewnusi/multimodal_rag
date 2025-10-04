import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import io

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import base64

from services.img_converter import ImageConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerationService:
    def __init__(self):
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            max_output_tokens=8192, 
        )

        self.image_converter = ImageConverter()

    def _encode_image_to_base64(self, image_path: str) -> Optional[Dict[str, str]]:
        try:
            converted = self.image_converter.convert_to_supported_format(image_path)
            
            if not converted:
                return None
            
            encoded = base64.b64encode(converted['data']).decode('utf-8')
            
            return {
                'base64': encoded,
                'mime_type': converted['mime_type']
            }
                
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def prepare_bidirectional_context(self, search_results: Dict[str, List[Dict[str, Any]]], 
                                n_articles: int = 3, n_images: int = 3) -> Dict[str, Any]:
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
            
            article_content = article.get('full_content', article.get('content', ''))
            
            context["primary_articles"].append({
                "title": article.get('title', ''),
                "content": article_content,
                "article_id": article_id,
                "url": article.get('url', '')
            })
        
        top_images = search_results.get('images', [])[:n_images]
        for image in top_images:
            article_id = image.get('article_id', '')
            
            source_article = None
            if article_id not in context["article_ids_covered"]:
                for article in search_results.get('text', []):
                    if article.get('article_id') == article_id:
                        source_article = {
                            "title": article.get('title', ''),
                            "article_id": article_id,
                            "url": article.get('url', '')
                        }
                        break
                
                if not source_article:
                    source_article = {
                        "title": image.get('title', 'Unknown article'),
                        "article_id": article_id,
                        "url": image.get('url', ''),
                        "is_placeholder": True
                    }
                
                context["article_ids_covered"].add(article_id)
            
            img_path = image.get('image_path', '')
            encoded_result = self._encode_image_to_base64(img_path)
            
            if encoded_result:
                context["primary_images"].append({
                    "filename": image.get('filename', Path(img_path).name),
                    "title": image.get('title', ''),
                    "article_id": article_id,
                    "source_article": source_article,
                    "image_path": img_path,
                    "base64": encoded_result['base64'],
                    "mime_type": encoded_result['mime_type']
                })
            else:
                logger.warning(f"Failed to encode image: {img_path}")
        
        context["total_unique_sources"] = len(context["article_ids_covered"])
        
        return context
    
    def format_bidirectional_context_for_prompt(self, context: Dict[str, Any]) -> str:
        formatted_context = []
        
        if context["primary_articles"]:
            formatted_context.append("PRIMARY ARTICLES (TEXT):")
            for i, article in enumerate(context["primary_articles"], 1):
                formatted_context.append(f"\n{i}. Article: {article['title']}")
                formatted_context.append(f"   Content: {article['content']}")
                if article['url']:
                    formatted_context.append(f"   URL: {article['url']}")
                formatted_context.append("")
        
        if context["primary_images"]:
            formatted_context.append("\nPRIMARY IMAGES WITH METADATA:")
            for i, image in enumerate(context["primary_images"], 1):
                formatted_context.append(f"\n{i}. Image: {image['filename']}")
                formatted_context.append(f"   From Article: {image['title']}")
                
                if image['source_article']:
                    source = image['source_article']
                    if source.get('url'):
                        formatted_context.append(f"   Article URL: {source['url']}")
                formatted_context.append("")
        
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
        try:
            context = self.prepare_bidirectional_context(search_results, n_articles, n_images)
            
            if context["total_unique_sources"] == 0:
                return "I couldn't find relevant information to answer your question. Please try rephrasing your query or asking about a different topic."
            
            formatted_context = self.format_bidirectional_context_for_prompt(context)
            
            message_content = []
            
            valid_images = []
            for img in context["primary_images"]:
                if img.get("base64"):
                    valid_images.append(img)
            
            valid_images = valid_images[:16]
            
            text_prompt = f"""Question: {query}

{formatted_context}

Please analyze the images I'm providing and give a comprehensive answer based on both the textual content and visual information."""
            
            message_content.append({
                "type": "text",
                "text": text_prompt
            })
            
            images_added = 0
            for img in valid_images:
                try:
                    if not img["base64"] or not img["mime_type"]:
                        logger.warning(f"Skipping image {img['filename']}: missing base64 or mime_type")
                        continue
                    
                    supported_types = ['image/jpeg', 'image/png', 'image/webp']
                    if img["mime_type"] not in supported_types:
                        logger.warning(f"Skipping image {img['filename']}: unsupported mime type {img['mime_type']}")
                        continue
                    
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['mime_type']};base64,{img['base64']}"
                        }
                    })
                    images_added += 1
                except Exception as img_error:
                    logger.error(f"Error adding image {img.get('filename', 'unknown')}: {img_error}")
                    continue
            
            logger.info(f"Successfully prepared {images_added} images for Gemini")
            
            messages = [
                SystemMessage(content=self.create_system_prompt()),
                HumanMessage(content=message_content)
            ]
            
            response = self.llm.invoke(
                messages,
                config={
                    'run_name': 'AnswerGeneration',
                    'metadata': {
                        'query': query, 
                        'n_articles': n_articles, 
                        'n_images': n_images, 
                        'images_added': images_added,
                        'article_ids': list(context["article_ids_covered"])
                    }
                }
            )
            
            logger.info(f"Generated answer using {context['total_unique_sources']} sources and {images_added} images")
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error while processing your question. Please try again."
    
    def generate_answer_with_summary(self, query: str, search_results: Dict[str, List[Dict[str, Any]]], 
                                   n_articles: int = 3, n_images: int = 3) -> Dict[str, Any]:
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
        enhanced_results = search_results.copy()
        
        text_article_ids = {result.get('article_id') for result in search_results.get('text', [])}
        
        if 'images' in search_results:
            related_images = []
            standalone_images = []
            
            for image in search_results['images']:
                if image.get('article_id') in text_article_ids:
                    related_images.append(image)
                else:
                    standalone_images.append(image)
            
            enhanced_results['images'] = related_images + standalone_images[:3]
        
        for text_result in enhanced_results.get('text', []):
            article_id = text_result.get('article_id')
            related_imgs = [img for img in enhanced_results.get('images', []) 
                          if img.get('article_id') == article_id]
            text_result['associated_images'] = related_imgs
            text_result['has_visual_content'] = len(related_imgs) > 0
        
        return enhanced_results


def main():
    pass

if __name__ == "__main__":
    main()