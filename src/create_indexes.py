import os
import numpy as np
import json
import shutil
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from services.text_vector_store_service import TextVectorStoreService
from services.image_vector_store_service import ImageVectorStoreService


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedIndexCreator:
    def __init__(self, data_folder: str = "data", indexes_folder: str = "indexes"):
        self.data_folder = Path(data_folder)
        self.indexes_folder = Path(indexes_folder)
        
        self.indexes_folder.mkdir(exist_ok=True)
        
        self._validate_data_folder()
    
    def _validate_data_folder(self):
        """Validate that data folder has required structure."""
        required_paths = [
            self.data_folder / "txt",
            self.data_folder / "img", 
            self.data_folder / "metadata.json"
        ]
        
        missing = [path for path in required_paths if not path.exists()]
        
        if missing:
            logger.error("Missing required data folder structure")
            sys.exit(1)
    
    def create_text_index(self, force_recreate: bool = False) -> bool:
        text_index_path = self.indexes_folder / "text_index"
        
        if text_index_path.exists() and not force_recreate:
            logger.info(f"Text index already exists at {text_index_path}. Use --force to recreate")
            return True
        
        try:            
            text_service = TextVectorStoreService(
                data_folder=str(self.data_folder),
                metadata_file="metadata.json"
            )
            
            documents = text_service.process_text_files()
            
            if not documents:
                logger.warning("No text documents found to process")
                return False
            
            logger.info(f"Creating text vector store with {len(documents)} documents...")
            vector_store = text_service.create_vector_store(documents)
            
            text_service.save_vector_store(vector_store, str(text_index_path))
            
            stats = text_service.get_stats(vector_store)
            
            logger.info("Text index creation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating text index: {e}")
            return False
    
    def create_image_index(self, force_recreate: bool = False) -> bool:
        image_index_path = self.indexes_folder / "image_index"
        
        if image_index_path.exists() and not force_recreate:
            logger.info(f"Image index already exists at {image_index_path}. Use --force to recreate")
            return True
        
        try:
            image_service = ImageVectorStoreService(
                data_folder=str(self.data_folder),
                metadata_file="metadata.json"
            )
            
            documents = image_service.process_image_files()
            
            if not documents:
                logger.warning("No image documents found to process")
                return False
            
            logger.info(f"Creating image vector store with {len(documents)} documents...")
            vector_store = image_service.create_vector_store(documents)
            
            image_service.save_vector_store(vector_store, str(image_index_path))
            
            stats = image_service.get_stats(vector_store)
            
            logger.info("Image index creation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating image index: {e}")
            return False

    
    def create_both_indexes(self, force_recreate: bool = False) -> bool:
        text_success = self.create_text_index(force_recreate)
        image_success = self.create_image_index(force_recreate)
        
        logger.info(f"Text index: {'SUCCESS' if text_success else 'FAILED'}")
        logger.info(f"Image index: {'SUCCESS' if image_success else 'FAILED'}")
        
        if text_success and image_success:
            logger.info("\nBoth indexes created successfully!")
            logger.info(f"Indexes saved to: {self.indexes_folder}")
            return True
        else:
            logger.error("\nFailed to create indexes")
            return False
    
    def check_environment(self) -> bool:
        """Check if environment is properly configured."""
        
        project_id = os.getenv("PROJECT_ID")
        if not project_id:
            logger.error("ERROR: PROJECT_ID not set in environment")
            return False
        
        txt_files = list((self.data_folder / "txt").glob("*.txt"))
        img_files = list((self.data_folder / "img").glob("*"))
        
        if not txt_files and not img_files:
            logger.warning("WARNING: No text or image files found")
            return False
        return True


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Create FAISS indexes for multimodal RAG")
    parser.add_argument("--data", default="data", help="Path to data folder")
    parser.add_argument("--indexes", default="indexes", help="Path to save indexes")
    parser.add_argument("--text-only", action="store_true", help="Create only text index")
    parser.add_argument("--image-only", action="store_true", help="Create only image index")
    parser.add_argument("--force", action="store_true", help="Force recreate existing indexes")
    parser.add_argument("--check", action="store_true", help="Only check environment")

    args = parser.parse_args()
    
    creator = UnifiedIndexCreator(args.data, args.indexes)
    
    if not creator.check_environment():
        logger.error("Environment check failed")
        sys.exit(1)
    
    if args.check:
        logger.info("Environment check completed successfully")
        return
    
    if args.text_only:
        success = creator.create_text_index(args.force)
    elif args.image_only:
        success = creator.create_image_index(args.force)
    else:
        success = creator.create_both_indexes(args.force)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()