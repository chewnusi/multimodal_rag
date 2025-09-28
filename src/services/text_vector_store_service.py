import os
import getpass
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import time
import random
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextVectorStoreService:
    """
    Service for creating text vector store from article files.
    """
    
    def __init__(self, data_folder: str = "data", metadata_file: str = "metadata.json"):
        """
        Initialize the text vector store service.
        
        Args:
            data_folder: Path to data folder containing txt/ and img/ subfolders
            metadata_file: Path to metadata.json file
        """
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.data_folder = Path(data_folder)
        self.txt_folder = self.data_folder / "txt"
        self.metadata_file = self.data_folder / metadata_file
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT" # QUESTION_ANSWERING
            )
        
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized TextVectorStoreService for folder: {data_folder}")
    
    def _load_metadata(self) -> dict:
        """Load metadata.json file."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {self.metadata_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing metadata: {e}")
            return {}
    
    def _extract_article_id_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract article ID from filename.
        
        Args:
            filename: Filename like "1_GPT_updates.txt"
            
        Returns:
            Article ID as string or None if extraction fails
        """
        try:
            return filename.split('_')[0]
        except (IndexError, ValueError):
            logger.warning(f"Could not extract article ID from filename: {filename}")
            return None
    
    def _read_text_file(self, file_path: Path) -> Optional[str]:
        """Read content from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def _create_document(self, article_id: str, content: str, filename: str) -> Optional[Document]:
        """
        Create a LangChain Document with metadata.
        
        Args:
            article_id: Article ID
            content: Article text content
            filename: Original filename
            
        Returns:
            Document object or None if metadata not found
        """
        if article_id not in self.metadata:
            logger.warning(f"No metadata found for article ID: {article_id}")
            return None
        
        article_meta = self.metadata[article_id]
        
        document = Document(
            page_content=content,
            metadata={
                "article_id": article_id,
                "title": article_meta.get("title", ""),
                "url": article_meta.get("url", ""),
                "date": article_meta.get("date", ""),
                "filename": filename
            }
        )
        
        return document
    
    def scan_text_files(self) -> List[Path]:
        """
        Scan txt folder for text files.
        
        Returns:
            List of text file paths
        """
        if not self.txt_folder.exists():
            logger.error(f"Text folder not found: {self.txt_folder}")
            return []
        
        txt_files = list(self.txt_folder.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files")
        
        return txt_files
    
    def process_text_files(self) -> List[Document]:
        """
        Process all text files and create documents.
        
        Returns:
            List of Document objects
        """
        txt_files = self.scan_text_files()
        documents = []
        
        for txt_file in txt_files:
            filename = txt_file.name
            article_id = self._extract_article_id_from_filename(filename)
            
            if not article_id:
                continue
            
            # Read file content
            content = self._read_text_file(txt_file)
            if not content:
                continue
            
            # Create document
            document = self._create_document(article_id, content, filename)
            if document:
                documents.append(document)
                logger.info(f"Processed article {article_id}: {document.metadata['title'][:50]}...")
        
        logger.info(f"Successfully processed {len(documents)} documents")
        return documents

    def retry_with_exponential_backoff(max_retries=3, base_delay=5, max_delay=60):
        """Decorator for exponential backoff retry."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        error_msg = str(e).lower()
                        
                        # Check if it's a quota/rate limit error
                        if any(keyword in error_msg for keyword in ['quota', '429', 'rate limit', 'exceeded']):
                            if attempt < max_retries:
                                # Calculate delay with exponential backoff + jitter
                                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                                logger.warning(f"Quota exceeded, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                                time.sleep(delay)
                                continue
                        
                        # Re-raise if not a quota error or max retries exceeded
                        raise e
                
                return None  # All retries failed
            return wrapper
        return decorator
    
    @retry_with_exponential_backoff(max_retries=7, base_delay=5, max_delay=60)
    def create_vector_store(self, documents: Optional[List[Document]] = None) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of documents. If None, processes all text files.
            
        Returns:
            FAISS vector store
        """
        if documents is None:
            documents = self.process_text_files()
        
        if not documents:
            raise ValueError("No documents to process")
        
        logger.info(f"Creating vector store with {len(documents)} documents...")
        
        try:
            # Process in smaller batches to avoid quota issues
            batch_size = 5  # Reduce batch size
            all_embeddings = []
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                logger.info(f"Processing document batch {i//batch_size + 1}")
                
                if i > 0:
                    time.sleep(10)
                
                vector_store = FAISS.from_documents(
                    documents=batch_docs,
                    embedding=self.embeddings
                )
                
                if i == 0:
                    main_vector_store = vector_store
                else:
                    main_vector_store.merge_from(vector_store)
            
            logger.info("Text vector store created successfully")
            return main_vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vector_store(self, vector_store: FAISS, save_path: str = "text_index") -> None:
        """
        Save vector store to disk.
        
        Args:
            vector_store: FAISS vector store to save
            save_path: Path to save the index
        """
        try:
            vector_store.save_local(save_path)
            logger.info(f"Vector store saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_vector_store(self, load_path: str = "text_index") -> FAISS:
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
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from: {load_path}")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def get_stats(self, vector_store: FAISS) -> Dict:
        """Get statistics about the vector store."""
        try:
            num_vectors = vector_store.index.ntotal
            
            sample_docs = vector_store.similarity_search("test", k=1)
            sample_metadata = sample_docs[0].metadata if sample_docs else {}
            
            return {
                "num_vectors": num_vectors,
                "embedding_dimension": vector_store.index.d,
                "sample_metadata_keys": list(sample_metadata.keys()),
                "index_type": type(vector_store.index).__name__
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


def main():
    # TODO: cleanup logs
    """Main function to create text vector store."""
    
    service = TextVectorStoreService(data_folder="data", metadata_file="metadata.json")
    
    try:
        print("Processing text files...")
        documents = service.process_text_files()
        
        if not documents:
            print("No documents found to process")
            return
        
        print(f"Creating vector store with {len(documents)} documents...")
        vector_store = service.create_vector_store(documents)
        
        stats = service.get_stats(vector_store)
        print(f"Vector store stats: {stats}")
        
        print("Saving vector store...")
        service.save_vector_store(vector_store, "text_index")
        
        print("Text vector store creation completed successfully")
        
        print("\nTesting search...")
        results = vector_store.similarity_search("AI technology", k=3)
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.metadata['title']} (ID: {doc.metadata['article_id']})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()