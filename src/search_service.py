import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from dotenv import load_dotenv
load_dotenv()

# Import your services
try:
    from services.text_vector_store_service import TextVectorStoreService
    from services.image_vector_store_service import ImageVectorStoreService
except ImportError:
    # Fallback for different directory structures
    try:
        from text_vector_store_service import TextVectorStoreService
        from image_vector_store_service import ImageVectorStoreService
    except ImportError as e:
        print(f"Error importing services: {e}")
        raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalSearchService:
    """
    Service for searching both text and image indexes.
    Handles loading, caching, and querying FAISS indexes.
    """
    
    def __init__(self, 
                 data_folder: str = None, 
                 indexes_folder: str = None):
        """
        Initialize the search service.
        
        Args:
            data_folder: Path to data folder
            indexes_folder: Path to indexes folder
        """
        # Auto-detect paths based on current working directory
        if data_folder is None:
            # Try different possible locations
            possible_data_paths = [
                Path("data"),     
                Path("../data"),     
            ]
            
            data_folder = None
            for path in possible_data_paths:
                if path.exists() and (path / "metadata.json").exists():
                    data_folder = str(path)
                    break
            
            if data_folder is None:
                logger.error("Could not find data folder. Please specify data_folder parameter.")
                data_folder = "data"  # fallback
        
        if indexes_folder is None:
            # Try different possible locations
            possible_index_paths = [
                Path("indexes"),
                Path("../indexes"),
                Path("text_index").parent if Path("text_index").exists() else None,
                Path("image_index").parent if Path("image_index").exists() else None
            ]
            
            indexes_folder = None
            for path in possible_index_paths:
                if path and path.exists():
                    if (path / "text_index").exists() or (path / "image_index").exists():
                        indexes_folder = str(path)
                        break
            
            if indexes_folder is None:
                logger.warning("Could not find indexes folder. Using default 'indexes'")
                indexes_folder = "indexes"
        
        self.data_folder = Path(data_folder)
        self.indexes_folder = Path(indexes_folder)
        
        # Initialize services
        self.text_service = None
        self.image_service = None
        
        # Cached vector stores
        self.text_vector_store = None
        self.image_vector_store = None
        
        # Load metadata for enriching results
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized MultimodalSearchService")
        logger.info(f"Data folder: {self.data_folder}")
        logger.info(f"Indexes folder: {self.indexes_folder}")
    
    def _load_metadata(self) -> Dict:
        """Load metadata for enriching search results."""
        try:
            metadata_file = self.data_folder / "metadata.json"
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            return {}
    
    def _ensure_text_service(self):
        """Lazy initialization of text service."""
        if self.text_service is None:
            self.text_service = TextVectorStoreService(
                data_folder=str(self.data_folder),
                metadata_file="metadata.json"
            )
    
    def _ensure_image_service(self):
        """Lazy initialization of image service."""
        if self.image_service is None:
            self.image_service = ImageVectorStoreService(
                data_folder=str(self.data_folder),
                metadata_file="metadata.json"
            )
    
    def load_text_index(self) -> bool:
        """
        Load text vector store from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.text_vector_store is not None:
                return True
                
            text_index_path = self.indexes_folder / "text_index"
            if not text_index_path.exists():
                logger.error(f"Text index not found: {text_index_path}")
                return False
            
            self._ensure_text_service()
            self.text_vector_store = self.text_service.load_vector_store(str(text_index_path))
            logger.info("Text index loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading text index: {e}")
            return False
    
    def load_image_index(self) -> bool:
        """
        Load image vector store from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.image_vector_store is not None:
                return True
                
            image_index_path = self.indexes_folder / "image_index"
            if not image_index_path.exists():
                logger.error(f"Image index not found: {image_index_path}")
                return False
            
            self._ensure_image_service()
            self.image_vector_store = self.image_service.load_vector_store(str(image_index_path))
            logger.info("Image index loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading image index: {e}")
            return False
    
    def load_all_indexes(self) -> Tuple[bool, bool]:
        """
        Load both text and image indexes.
        
        Returns:
            Tuple of (text_success, image_success)
        """
        text_success = self.load_text_index()
        image_success = self.load_image_index()
        return text_success, image_success
    
    def search_text(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search text documents and include associated images.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with metadata and associated images
        """
        try:
            if not self.load_text_index():
                return []
            
            # Perform search
            results = self.text_vector_store.similarity_search(query, k=k)
            
            # Format results
            formatted_results = []
            for doc in results:
                # Find associated images for this article
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
                    "images": associated_images  # Add images list
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return []
    
    def _find_images_for_article(self, article_id: str) -> List[str]:
        """
        Find image files associated with an article ID.
        
        Args:
            article_id: Article ID to search for
            
        Returns:
            List of image file paths
        """
        if not article_id:
            return []
        
        try:
            img_folder = self.data_folder / "img"
            if not img_folder.exists():
                return []
            
            # Look for images that start with the article ID
            pattern = f"{article_id}_*"
            image_files = []
            
            # Check common image extensions
            for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                matches = list(img_folder.glob(f"{pattern}{ext}"))
                image_files.extend([str(img) for img in matches])
            
            return image_files
            
        except Exception as e:
            logger.error(f"Error finding images for article {article_id}: {e}")
            return []
    
    def search_images(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search image documents using text query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            if not self.load_image_index():
                return []
            
            # Perform search
            results = self.image_vector_store.similarity_search(query, k=k)
            
            # Format results
            formatted_results = []
            for doc in results:
                result = {
                    "type": "image",
                    "image_path": doc.page_content,  # Image path is stored as content
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
        """
        Search both text and images simultaneously.
        
        Args:
            query: Search query
            k_text: Number of text results
            k_images: Number of image results
            
        Returns:
            Dictionary with 'text' and 'images' keys containing results
        """
        text_results = self.search_text(query, k_text)
        image_results = self.search_images(query, k_images)
        
        return {
            "text": text_results,
            "images": image_results
        }
    
    def get_full_article(self, article_id: str) -> Optional[str]:
        """
        Get full article content by ID.
        
        Args:
            article_id: Article ID
            
        Returns:
            Full article content or None if not found
        """
        try:
            # Find the text file
            txt_files = list((self.data_folder / "txt").glob(f"{article_id}_*.txt"))
            if not txt_files:
                return None
            
            with open(txt_files[0], 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error reading article {article_id}: {e}")
            return None
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded indexes."""
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