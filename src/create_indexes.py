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
            print("Missing required data folder structure:")
            for path in missing:
                print(f"  - {path}")
            print("\nExpected structure:")
            print("  data/")
            print("    ├── txt/")
            print("    ├── img/")
            print("    └── metadata.json")
            sys.exit(1)
    
    def create_text_index(self, force_recreate: bool = False) -> bool:
        text_index_path = self.indexes_folder / "text_index"
        
        if text_index_path.exists() and not force_recreate:
            print(f"Text index already exists at {text_index_path}")
            print("Use --force to recreate")
            return True
        
        try:
            print("\n" + "="*50)
            print("CREATING TEXT INDEX")
            print("="*50)
            
            text_service = TextVectorStoreService(
                data_folder=str(self.data_folder),
                metadata_file="metadata.json"
            )
            
            print("Processing text files...")
            documents = text_service.process_text_files()
            
            if not documents:
                print("No text documents found to process")
                return False
            
            print(f"Creating text vector store with {len(documents)} documents...")
            vector_store = text_service.create_vector_store(documents)
            
            print("Saving text index...")
            text_service.save_vector_store(vector_store, str(text_index_path))
            
            stats = text_service.get_stats(vector_store)
            print(f"Text index stats: {stats}")
            
            print("Text index creation completed successfully")
            return True
            
        except Exception as e:
            print(f"Error creating text index: {e}")
            return False
    
    def create_image_index(self, force_recreate: bool = False) -> bool:
        image_index_path = self.indexes_folder / "image_index"
        
        if image_index_path.exists() and not force_recreate:
            print(f"Image index already exists at {image_index_path}")
            print("Use --force to recreate")
            return True
        
        try:
            print("\n" + "="*50)
            print("CREATING IMAGE INDEX")
            print("="*50)
            
            image_service = ImageVectorStoreService(
                data_folder=str(self.data_folder),
                metadata_file="metadata.json"
            )
            
            print("Processing image files...")
            documents = image_service.process_image_files()
            
            if not documents:
                print("No image documents found to process")
                return False
            
            print(f"Creating image vector store with {len(documents)} documents...")
            vector_store = image_service.create_vector_store(documents)
            
            print("Saving image index...")
            image_service.save_vector_store(vector_store, str(image_index_path))
            
            stats = image_service.get_stats(vector_store)
            print(f"Image index stats: {stats}")
            
            print("Image index creation completed successfully")
            return True
            
        except Exception as e:
            print(f"Error creating image index: {e}")
            return False

    
    def create_both_indexes(self, force_recreate: bool = False) -> bool:
        text_success = self.create_text_index(force_recreate)
        image_success = self.create_image_index(force_recreate)
        
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Text index: {'SUCCESS' if text_success else 'FAILED'}")
        print(f"Image index: {'SUCCESS' if image_success else 'FAILED'}")
        
        if text_success and image_success:
            print("\nBoth indexes created successfully!")
            print(f"Indexes saved to: {self.indexes_folder}")
            return True
        else:
            print("\nFailed to create indexes")
            return False
    
    def check_environment(self) -> bool:
        """Check if environment is properly configured."""
        print("Checking environment...")
        
        project_id = os.getenv("PROJECT_ID")
        if not project_id:
            print("ERROR: PROJECT_ID not set in environment")
            return False
        
        txt_files = list((self.data_folder / "txt").glob("*.txt"))
        img_files = list((self.data_folder / "img").glob("*"))
        
        print(f"Text files found: {len(txt_files)}")
        print(f"Image files found: {len(img_files)}")
        
        if not txt_files and not img_files:
            print("WARNING: No text or image files found")
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
        print("Environment check failed")
        sys.exit(1)
    
    if args.check:
        print("Environment check completed successfully")
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