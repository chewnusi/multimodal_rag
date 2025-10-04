import os
import base64
import json
import logging
from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
import io
import time
import random
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

from google.cloud import aiplatform
from google.protobuf import struct_pb2
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image

from langchain_core.embeddings import Embeddings

from utils import (
    load_metadata,
    extract_article_id_from_filename,
    get_contextual_text_from_path,
    retry_with_exponential_backoff
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEmbeddingService(Embeddings):
    SUPPORTED_FORMATS = {'.bmp', '.gif', '.jpg', '.png'}
    CONVERTIBLE_FORMATS = {'.jpeg', '.webp', '.tiff', '.tif', '.ico'}
    
    def __init__(self, project_id: str, metadata_file: str = "metadata.json"):
        self.DIMENSION = 1408
        self.metadata_file = Path(metadata_file) if Path(metadata_file).is_absolute() else Path(metadata_file)
        self.metadata = load_metadata(self.metadata_file)

        self.project_id = project_id

        try:
            vertexai.init(project=project_id)
            
            if not os.getenv('GOOGLE_CLOUD_QUOTA_PROJECT'):
                os.environ['GOOGLE_CLOUD_QUOTA_PROJECT'] = project_id
            
            self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
            
            logger.info(f"Initialized ImageEmbeddingService for project: {project_id}")
        
        except Exception as e:
            logger.error(f"Failed to initialize service: {str(e)}")
            logger.error("Make sure:")
            logger.error("1. Vertex AI API is enabled: gcloud services enable aiplatform.googleapis.com")
            logger.error("2. Quota project is set: gcloud auth application-default set-quota-project YOUR_PROJECT_ID")
            raise



    def _is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.SUPPORTED_FORMATS
    
    def _is_convertible_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is convertible to supported format."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.CONVERTIBLE_FORMATS
    
    def _convert_image_to_supported_format(self, image_path: Union[str, Path]) -> Optional[bytes]:
        try:
            input_suffix = Path(image_path).suffix.lower()
            
            with Image.open(image_path) as img:
                if input_suffix == '.jpeg' and img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, 'white')
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if hasattr(img, 'split') and len(img.split()) > 3 else None)
                    img = background
                elif img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB' if input_suffix == '.jpeg' else 'RGBA')
                
                img_buffer = io.BytesIO()
                format_name = 'JPEG' if input_suffix == '.jpeg' else 'PNG'
                save_kwargs = {'optimize': True}
                if format_name == 'JPEG':
                    save_kwargs['quality'] = 95
                
                img.save(img_buffer, format=format_name, **save_kwargs)
                return img_buffer.getvalue()
                
        except Exception as e:
            logger.error(f"PIL conversion failed for {image_path}: {str(e)}")
            return None
    
    def _encode_image_to_base64(self, image_data: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def _load_image_file(self, image_path: Union[str, Path]) -> Optional[bytes]:
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            if self._is_supported_format(image_path):
                with open(image_path, 'rb') as f:
                    return f.read()
            
            elif self._is_convertible_format(image_path):
                logger.info(f"Converting {image_path.suffix} to supported format: {image_path}")
                return self._convert_image_to_supported_format(image_path)
            
            else:
                logger.warning(f"Unsupported image format: {image_path.suffix} for file {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None



    @retry_with_exponential_backoff(max_retries=5, base_delay=10, max_delay=90)
    def generate_embedding_from_path(self, image_path: Union[str, Path], 
                                   contextual_text: str = "") -> Optional[np.ndarray]:
        try:
            image_bytes = self._load_image_file(image_path)
            if image_bytes is None:
                return None
            
            image = Image.load_from_file(str(image_path))
            
            embeddings = self.model.get_embeddings(
                image=image,
                contextual_text=contextual_text,
                dimension=self.DIMENSION
            )
            
            if embeddings.image_embedding:
                logger.info(f"Generated embedding for: {image_path}")
                return np.array(embeddings.image_embedding)
            else:
                logger.error(f"No embedding returned for: {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {str(e)}")
            return None
    

    def process_image_batch(self, image_paths: List[Union[str, Path]], 
                          contextual_texts: Optional[List[str]] = None) -> List[Tuple[str, Optional[np.ndarray]]]:
        if contextual_texts is None:
            contextual_texts = [""] * len(image_paths)
        
        if len(contextual_texts) != len(image_paths):
            raise ValueError("contextual_texts length must match image_paths length")
        
        results = []
        total_images = len(image_paths)
        
        logger.info(f"Processing batch of {total_images} images")
        
        for i, (image_path, context) in enumerate(zip(image_paths, contextual_texts)):
            logger.info(f"Processing {i+1}/{total_images}: {image_path}")
            
            embedding = self.generate_embedding_from_path(image_path, context)
            results.append((str(image_path), embedding))
        
        successful = sum(1 for _, emb in results if emb is not None)
        logger.info(f"Batch processing complete: {successful}/{total_images} successful")
        
        return results

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        
        for image_path in texts:
            contextual_text = get_contextual_text_from_path(image_path, self.metadata)
            
            embedding = self.generate_embedding_from_path(image_path, contextual_text=contextual_text)
            if embedding is not None:
                embeddings.append(embedding.tolist())
            else:
                embeddings.append([0.0] * self.DIMENSION)
                logger.warning(f"Failed to embed image {image_path}, using zero vector")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        if self._is_image_path(text):
            embedding = self.generate_embedding_from_path(text, contextual_text="")
            if embedding is not None:
                return embedding.tolist()
            else:
                logger.warning(f"Failed to embed image query {text}, using zero vector")
                return [0.0] * self.DIMENSION
        else:
            try:
                embeddings = self.model.get_embeddings(
                    contextual_text=text,
                    dimension=self.DIMENSION
                )
                
                if embeddings.text_embedding:
                    return np.array(embeddings.text_embedding).tolist()
                else:
                    logger.warning(f"Failed to embed text query '{text}', using zero vector")
                    return [0.0] * self.DIMENSION

            except Exception as e:
                logger.error(f"Error embedding text query '{text}': {str(e)}")
                return [0.0] * self.DIMENSION

    def _is_image_path(self, text: str) -> bool:
        try:
            path = Path(text)
            
            if path.suffix.lower() in (self.SUPPORTED_FORMATS | self.CONVERTIBLE_FORMATS):
                return True
            
            if path.exists() and path.is_file():
                return self._is_supported_format(path) or self._is_convertible_format(path)
                
        except (OSError, ValueError):
            pass
        
        return False

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)
    
    def scan_directory(self, directory_path: Union[str, Path], 
                      recursive: bool = True) -> List[Path]:
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found or not a directory: {directory_path}")
            return []
        
        pattern = "**/*" if recursive else "*"
        all_files = directory_path.glob(pattern)
        
        supported_files = []
        convertible_files = []
        skipped_files = []
        
        for file_path in all_files:
            if file_path.is_file():
                if self._is_supported_format(file_path):
                    supported_files.append(file_path)
                elif self._is_convertible_format(file_path):
                    convertible_files.append(file_path)
                else:
                    suffix = file_path.suffix.lower()
                    skipped_files.append(file_path)
        
        logger.info(f"Directory scan results:")
        logger.info(f"  Supported formats: {len(supported_files)} files")
        logger.info(f"  Convertible formats: {len(convertible_files)} files")
        logger.info(f"  Skipped files: {len(skipped_files)} files")
        
        return supported_files + convertible_files


def main():
    pass

if __name__ == "__main__":
    main()