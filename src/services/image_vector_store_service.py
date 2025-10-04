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
from utils import (
    load_metadata,
    extract_article_id_from_filename,
    create_image_document
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageVectorStoreService:
    def __init__(self, data_folder: str = "data", metadata_file: str = "metadata.json"):
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
        
        self.metadata = load_metadata(self.metadata_file)

    def _is_zero_vector(self, vector: List[float]) -> bool:
        return all(val == 0.0 for val in vector)
    

    
    def scan_image_files(self) -> List[Path]:
        if not self.img_folder.exists():
            logger.error(f"Image folder not found: {self.img_folder}")
            return []
        
        image_files = self.image_embeddings.scan_directory(self.img_folder, recursive=False)
        return image_files
    
    def process_image_files(self) -> List[Document]:
        image_files = self.scan_image_files()
        documents = []
        
        for image_file in image_files:
            filename = image_file.name
            article_id = extract_article_id_from_filename(filename)
            
            if not article_id:
                continue
            
            document = create_image_document(article_id, image_file, self.metadata)
            if document:
                documents.append(document)
        
        logger.info(f"Processed {len(documents)} image documents")
        return documents

    def create_vector_store(self, documents: Optional[List[Document]] = None) -> FAISS:
        """
        Create FAISS vector store from image documents.
        Filters out documents with zero embeddings (failed image processing).
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
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{ (len(documents) + batch_size - 1) // batch_size }")
            
            if i > 0:
                time.sleep(15)
            
            batch_paths = [doc.page_content for doc in batch_docs]
            batch_embeddings = self.image_embeddings.embed_documents(batch_paths)
            
            for doc, embedding in zip(batch_docs, batch_embeddings):
                if self._is_zero_vector(embedding):
                    filtered_count += 1
                    logger.warning(f"Filtering out document with zero embedding: {doc.metadata['filename']}")
                else:
                    valid_texts.append(doc.page_content)
                    valid_embeddings.append(embedding)
                    valid_metadatas.append(doc.metadata)
        
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count} documents with zero embeddings")
        
        if not valid_texts:
            raise ValueError("No valid documents remaining after filtering zero embeddings")
        
        text_embedding_pairs = list(zip(valid_texts, valid_embeddings))
        
        try:
            logger.info(f"Creating FAISS index from {len(text_embedding_pairs)} pre-computed embeddings...")
            
            vector_store = FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=self.image_embeddings,
                metadatas=valid_metadatas
            )
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store from embeddings: {e}")
            raise
    
    def save_vector_store(self, vector_store: FAISS, save_path: str = "image_index") -> None:
        try:
            vector_store.save_local(save_path)
            logger.info(f"Image vector store saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving image vector store: {e}")
            raise
    
    def load_vector_store(self, load_path: str = "image_index") -> FAISS:
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
        return vector_store.similarity_search(query, k=k)
    
    def search_by_image(self, vector_store: FAISS, image_path: str, k: int = 5) -> List[Document]:
        return vector_store.similarity_search(image_path, k=k)


def main():
    pass

if __name__ == "__main__":
    main()