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

from utils import (
    load_metadata,
    extract_article_id_from_filename,
    create_text_document,
    retry_with_exponential_backoff,
    safe_read_file
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextVectorStoreService:
    
    def __init__(self, data_folder: str = "data", metadata_file: str = "metadata.json"):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.data_folder = Path(data_folder)
        self.txt_folder = self.data_folder / "txt"
        self.metadata_file = self.data_folder / metadata_file
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT"
            )
        
        self.metadata = load_metadata(self.metadata_file)
        

    
    def scan_text_files(self) -> List[Path]:
        if not self.txt_folder.exists():
            logger.error(f"Text folder not found: {self.txt_folder}")
            return []
        
        txt_files = list(self.txt_folder.glob("*.txt"))
        return txt_files
    
    def process_text_files(self) -> List[Document]:
        txt_files = self.scan_text_files()
        documents = []
        
        for txt_file in txt_files:
            filename = txt_file.name
            article_id = extract_article_id_from_filename(filename)
            
            if not article_id:
                continue
            
            content = safe_read_file(txt_file)
            if not content:
                continue
            
            document = create_text_document(article_id, content, filename, self.metadata)
            if document:
                documents.append(document)
        
        logger.info(f"Processed {len(documents)} text documents")
        return documents


    
    @retry_with_exponential_backoff(max_retries=7, base_delay=5, max_delay=60)
    def create_vector_store(self, documents: Optional[List[Document]] = None) -> FAISS:
        if documents is None:
            documents = self.process_text_files()
        
        if not documents:
            raise ValueError("No documents to process")
        
        logger.info(f"Creating vector store with {len(documents)} documents...")
        
        try:
            batch_size = 5
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
            
            return main_vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vector_store(self, vector_store: FAISS, save_path: str = "text_index") -> None:
        try:
            vector_store.save_local(save_path)
            logger.info(f"Vector store saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_vector_store(self, load_path: str = "text_index") -> FAISS:
        try:
            vector_store = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def get_stats(self, vector_store: FAISS) -> Dict:
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
    pass

if __name__ == "__main__":
    main()