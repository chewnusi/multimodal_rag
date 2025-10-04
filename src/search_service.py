import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from dotenv import load_dotenv
load_dotenv()

from services.text_vector_store_service import TextVectorStoreService
from services.image_vector_store_service import ImageVectorStoreService
from utils import load_metadata


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalSearchService:
    
    def __init__(self, 
                 data_folder: str = None, 
                 indexes_folder: str = None):
        if data_folder is None:
            data_folder = "data"
        
        if indexes_folder is None:
            indexes_folder = "indexes"
        
        self.data_folder = Path(data_folder)
        self.indexes_folder = Path(indexes_folder)
        
        self.text_service = None
        self.image_service = None
        
        self.text_vector_store = None
        self.image_vector_store = None
        
        metadata_file = self.data_folder / "metadata.json"
        self.metadata = load_metadata(metadata_file)
    
    def _ensure_text_service(self):
        if self.text_service is None:
            self.text_service = TextVectorStoreService(
                data_folder=str(self.data_folder),
                metadata_file="metadata.json"
            )
    
    def _ensure_image_service(self):
        if self.image_service is None:
            self.image_service = ImageVectorStoreService(
                data_folder=str(self.data_folder),
                metadata_file="metadata.json"
            )
    
    def load_text_index(self) -> bool:
        try:
            if self.text_vector_store is not None:
                return True
                
            text_index_path = self.indexes_folder / "text_index"
            if not text_index_path.exists():
                logger.error(f"Text index not found: {text_index_path}")
                return False
            
            self._ensure_text_service()
            self.text_vector_store = self.text_service.load_vector_store(str(text_index_path))
            return True
            
        except Exception as e:
            logger.error(f"Error loading text index: {e}")
            return False
    
    def load_image_index(self) -> bool:
        try:
            if self.image_vector_store is not None:
                return True
                
            image_index_path = self.indexes_folder / "image_index"
            if not image_index_path.exists():
                logger.error(f"Image index not found: {image_index_path}")
                return False
            
            self._ensure_image_service()
            self.image_vector_store = self.image_service.load_vector_store(str(image_index_path))
            return True
            
        except Exception as e:
            logger.error(f"Error loading image index: {e}")
            return False
    
    def load_all_indexes(self) -> Tuple[bool, bool]:
        text_success = self.load_text_index()
        image_success = self.load_image_index()
        return text_success, image_success
    
    def search_text(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.load_text_index():
                return []
            
            results = self.text_vector_store.similarity_search(query, k=k)
            
            formatted_results = []
            for doc in results:
                article_id = doc.metadata.get("article_id", "")
                associated_images = self._find_images_for_article(article_id)
                
                result = {
                    "type": "text",
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "full_content": doc.page_content,
                    "metadata": doc.metadata,
                    "article_id": article_id,
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("url", ""),
                    "date": doc.metadata.get("date", ""),
                    "filename": doc.metadata.get("filename", ""),
                    "images": associated_images
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return []
    
    def _find_images_for_article(self, article_id: str) -> List[str]:
        if not article_id:
            return []
        
        try:
            img_folder = self.data_folder / "img"
            if not img_folder.exists():
                return []
            
            pattern = f"{article_id}_*"
            image_files = []
            
            for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                matches = list(img_folder.glob(f"{pattern}{ext}"))
                image_files.extend([str(img) for img in matches])
            
            return image_files
            
        except Exception as e:
            logger.error(f"Error finding images for article {article_id}: {e}")
            return []
    
    def search_images(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.load_image_index():
                return []
            
            results = self.image_vector_store.similarity_search(query, k=k)
            
            formatted_results = []
            for doc in results:
                result = {
                    "type": "image",
                    "image_path": doc.page_content,
                    "metadata": doc.metadata,
                    "article_id": doc.metadata.get("article_id", ""),
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("url", ""),
                    "date": doc.metadata.get("date", ""),
                    "filename": doc.metadata.get("filename", "")
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching images: {e}")
            return []
    
    def search_multimodal(self, query: str, k_text: int = 3, k_images: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        text_results = self.search_text(query, k_text)
        image_results = self.search_images(query, k_images)
        
        return {
            "text": text_results,
            "images": image_results
        }
    
    def get_full_article(self, article_id: str) -> Optional[str]:
        try:
            txt_files = list((self.data_folder / "txt").glob(f"{article_id}_*.txt"))
            if not txt_files:
                return None
            
            with open(txt_files[0], 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error reading article {article_id}: {e}")
            return None
    
    def get_index_stats(self) -> Dict[str, Any]:
        stats = {}
        
        if self.text_vector_store:
            try:
                stats["text"] = {
                    "loaded": True,
                    "num_documents": self.text_vector_store.index.ntotal,
                    "embedding_dimension": self.text_vector_store.index.d
                }
            except:
                stats["text"] = {"loaded": True, "error": "Could not get stats"}
        else:
            stats["text"] = {"loaded": False}
        
        if self.image_vector_store:
            try:
                stats["images"] = {
                    "loaded": True,
                    "num_documents": self.image_vector_store.index.ntotal,
                    "embedding_dimension": self.image_vector_store.index.d
                }
            except:
                stats["images"] = {"loaded": True, "error": "Could not get stats"}
        else:
            stats["images"] = {"loaded": False}
        
        return stats