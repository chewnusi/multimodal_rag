import os
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import time
import random
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

from .img_embed_service import ImageEmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageVectorStoreService:
    """
    Service for creating image vector store from image files.
    """
    
    def __init__(self, data_folder: str = "data", metadata_file: str = "metadata.json"):
        """
        Initialize the image vector store service.
        
        Args:
            data_folder: Path to data folder containing txt/ and img/ subfolders
            metadata_file: Path to metadata.json file
        """
        self.data_folder = Path(data_folder)
        self.img_folder = self.data_folder / "img"
        self.metadata_file = self.data_folder / metadata_file
        
        PROJECT_ID = os.getenv("PROJECT_ID")
        if not PROJECT_ID:
            raise ValueError("PROJECT_ID not found in environment")
        
        self.image_embeddings = ImageEmbeddingService(
            project_id=PROJECT_ID,
            metadata_file=str(self.metadata_file)
        )
        
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized ImageVectorStoreService for folder: {data_folder}")
    
    def _load_metadata(self) -> Dict:
        """Load metadata.json file."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(metadata)} articles")
            return metadata
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {self.metadata_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing metadata: {e}")
            return {}
    
    def _extract_article_id_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract article ID from image filename.
        
        Args:
            filename: Filename like "1_GPT_updates.png"
            
        Returns:
            Article ID as string or None if extraction fails
        """
        try:
            return filename.split('_')[0]
        except (IndexError, ValueError):
            logger.warning(f"Could not extract article ID from filename: {filename}")
            return None

    def _is_zero_vector(self, vector: List[float]) -> bool:
        """
        Check if a vector contains all zeros (indicating embedding failure).
        
        Args:
            vector: Embedding vector to check
            
        Returns:
            True if vector is all zeros, False otherwise
        """
        return all(val == 0.0 for val in vector)
    
    def _create_document(self, article_id: str, image_path: Path) -> Optional[Document]:
        """
        Create a LangChain Document for an image with metadata.
        
        Args:
            article_id: Article ID
            image_path: Path to image file
            
        Returns:
            Document object or None if metadata not found
        """
        if article_id not in self.metadata:
            logger.warning(f"No metadata found for article ID: {article_id}")
            return None
        
        article_meta = self.metadata[article_id]
        
        document = Document(
            page_content=str(image_path), 
            metadata={
                "article_id": article_id,
                "title": article_meta.get("title", ""),
                "url": article_meta.get("url", ""),
                "date": article_meta.get("date", ""),
                "filename": image_path.name,
            }
        )
        
        return document
    
    def scan_image_files(self) -> List[Path]:
        """
        Scan img folder for supported image files.
        
        Returns:
            List of image file paths
        """
        if not self.img_folder.exists():
            logger.error(f"Image folder not found: {self.img_folder}")
            return []
        
        image_files = self.image_embeddings.scan_directory(self.img_folder, recursive=False)
        logger.info(f"Found {len(image_files)} image files")
        
        return image_files
    
    def process_image_files(self) -> List[Document]:
        """
        Process all image files and create documents.
        
        Returns:
            List of Document objects
        """
        image_files = self.scan_image_files()
        documents = []
        
        for image_file in image_files:
            filename = image_file.name
            article_id = self._extract_article_id_from_filename(filename)
            
            if not article_id:
                continue
            
            document = self._create_document(article_id, image_file)
            if document:
                documents.append(document)
                logger.info(f"Processed image {article_id}: {document.metadata['title'][:50]}...")
        
        logger.info(f"Successfully processed {len(documents)} image documents")
        return documents

    def create_vector_store(self, documents: Optional[List[Document]] = None) -> FAISS:
        """
        Create FAISS vector store from image documents.
        Filters out documents with zero embeddings (failed image processing).
        
        Args:
            documents: List of documents. If None, processes all image files.
            
        Returns:
            FAISS vector store
        """
        if documents is None:
            documents = self.process_image_files()
        
        if not documents:
            raise ValueError("No image documents to process")
        
        valid_texts = []
        valid_embeddings = []
        valid_metadatas = []
        filtered_count = 0
        batch_size = 2  
        
        logger.info("Generating embeddings in batches and filtering...")
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{ (len(documents) + batch_size - 1) // batch_size }")
            
            if i > 0:
                time.sleep(15) # Keep your delay logic
            
            batch_paths = [doc.page_content for doc in batch_docs]
            batch_embeddings = self.image_embeddings.embed_documents(batch_paths)
            
            for doc, embedding in zip(batch_docs, batch_embeddings):
                if self._is_zero_vector(embedding):
                    filtered_count += 1
                    logger.warning(f"Filtering out document with zero embedding: {doc.metadata['filename']}")
                else:
                    # Store the successful results
                    valid_texts.append(doc.page_content)
                    valid_embeddings.append(embedding)
                    valid_metadatas.append(doc.metadata)
        
        logger.info(f"Filtered out {filtered_count} documents with zero embeddings")
        logger.info(f"Proceeding with {len(valid_texts)} valid documents")
        
        if not valid_texts:
            raise ValueError("No valid documents remaining after filtering zero embeddings")
        
        # Combine texts and their embeddings into the required format
        text_embedding_pairs = list(zip(valid_texts, valid_embeddings))
        
        try:
            logger.info(f"Creating FAISS index from {len(text_embedding_pairs)} pre-computed embeddings...")
            
            # Use FAISS.from_embeddings which does NOT call the embedding service again
            vector_store = FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=self.image_embeddings, # The embedding function is still needed for later searches
                metadatas=valid_metadatas
            )
            
            logger.info("Image vector store created successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store from embeddings: {e}")
            raise
    
    def save_vector_store(self, vector_store: FAISS, save_path: str = "image_index") -> None:
        """
        Save vector store to disk.
        
        Args:
            vector_store: FAISS vector store to save
            save_path: Path to save the index
        """
        try:
            vector_store.save_local(save_path)
            logger.info(f"Image vector store saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving image vector store: {e}")
            raise
    
    def load_vector_store(self, load_path: str = "image_index") -> FAISS:
        """
        Load vector store from disk.
        
        Args:
            load_path: Path to load the index from
            
        Returns:
            Loaded FAISS vector store
        """
        try:
            vector_store = FAISS.load_local(
                load_path, 
                self.image_embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Image vector store loaded from: {load_path}")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading image vector store: {e}")
            raise
    
    def get_stats(self, vector_store: FAISS) -> Dict:
        """Get statistics about the image vector store."""
        try:
            num_vectors = vector_store.index.ntotal
            
            sample_docs = vector_store.similarity_search("technology", k=1)
            sample_metadata = sample_docs[0].metadata if sample_docs else {}
            
            return {
                "num_vectors": num_vectors,
                "embedding_dimension": vector_store.index.d,
                "sample_metadata_keys": list(sample_metadata.keys()),
                "index_type": type(vector_store.index).__name__,
                "content_type": "image"
            }
        except Exception as e:
            logger.error(f"Error getting image vector store stats: {e}")
            return {}
    
    def search_by_text(self, vector_store: FAISS, query: str, k: int = 5) -> List[Document]:
        """
        Search images using text query.
        
        Args:
            vector_store: FAISS vector store
            query: Text query
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        return vector_store.similarity_search(query, k=k)
    
    def search_by_image(self, vector_store: FAISS, image_path: str, k: int = 5) -> List[Document]:
        """
        Search images using another image.
        
        Args:
            vector_store: FAISS vector store
            image_path: Path to query image
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        return vector_store.similarity_search(image_path, k=k)


def main():
    """Main function to create image vector store."""
    
    # Initialize service
    service = ImageVectorStoreService(data_folder="data", metadata_file="metadata.json")
    
    try:
        # Process image files and create vector store
        print("Processing image files...")
        documents = service.process_image_files()
        
        if not documents:
            print("No image documents found to process")
            return
        
        print(f"Creating image vector store with {len(documents)} documents...")
        vector_store = service.create_vector_store(documents)
        
        # Get and display stats
        stats = service.get_stats(vector_store)
        print(f"Image vector store stats: {stats}")
        
        # Save vector store
        print("Saving image vector store...")
        service.save_vector_store(vector_store, "image_index")
        
        print("Image vector store creation completed successfully")
        
        # Test searches
        print("\nTesting text-to-image search...")
        text_results = service.search_by_text(vector_store, "artificial intelligence", k=3)
        for i, doc in enumerate(text_results):
            print(f"{i+1}. {doc.metadata['title']} - {doc.metadata['filename']}")
        
        print("\nTesting image-to-image search...")
        if documents:
            sample_image = documents[0].metadata["filename"]
            sample_image = service.img_folder / sample_image
            image_results = service.search_by_image(vector_store, sample_image, k=3)
            for i, doc in enumerate(image_results):
                print(f"{i+1}. {doc.metadata['title']} - {doc.metadata['filename']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()