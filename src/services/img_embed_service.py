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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEmbeddingService(Embeddings):
    """
    Service for generating image embeddings using Google's multimodal embedding model.
    Supports BMP, GIF, JPG, PNG formats with automatic conversion capabilities.
    """
    
    SUPPORTED_FORMATS = {'.bmp', '.gif', '.jpg', '.png'}
    CONVERTIBLE_FORMATS = {'.jpeg', '.webp', '.tiff', '.tif', '.ico'}
    
    def __init__(self, project_id: str, metadata_file: str = "metadata.json"):
        """
        Initialize the Image Embedding Service.
        
        Args:
            project_id: Google Cloud project ID
        """
        self.DIMENSION = 1408  # Embedding dimension for multimodalembedding@001
        self.metadata_file = Path(metadata_file) if Path(metadata_file).is_absolute() else Path(metadata_file)
        self.metadata = self._load_metadata()

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

    def _load_metadata(self) -> dict:
        """Load metadata.json file for contextual text lookup."""
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {self.metadata_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing metadata: {e}")
            return {}

    def _get_contextual_text_from_path(self, image_path: str) -> str:
        """Extract contextual text (title) from metadata based on image filename."""
        try:
            filename = Path(image_path).name
            img_id = filename.split('_')[0]
            
            if img_id in self.metadata:
                return self.metadata[img_id]['title']
            else:
                logger.warning(f"No metadata found for image ID: {img_id}")
                return ""
        except (IndexError, ValueError, KeyError) as e:
            logger.warning(f"Could not extract contextual text for {image_path}: {e}")
            return ""

    def _is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.SUPPORTED_FORMATS
    
    def _is_convertible_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is convertible to supported format."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.CONVERTIBLE_FORMATS
    
    def _convert_image_to_supported_format(self, image_path: Union[str, Path]) -> Optional[bytes]:
        """
        Convert image using PIL - simplified version.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Converted image bytes or None if conversion fails
        """
        try:
            input_suffix = Path(image_path).suffix.lower()
            
            with Image.open(image_path) as img:
                # Handle transparency for JPEG conversion
                if input_suffix == '.jpeg' and img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, 'white')
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if hasattr(img, 'split') and len(img.split()) > 3 else None)
                    img = background
                elif img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB' if input_suffix == '.jpeg' else 'RGBA')
                
                # Save with appropriate format
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
        """
        Load image file and return bytes.
        Handles conversion if necessary.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image bytes or None if loading fails
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            # Check if format is directly supported
            if self._is_supported_format(image_path):
                with open(image_path, 'rb') as f:
                    return f.read()
            
            # Check if format is convertible
            elif self._is_convertible_format(image_path):
                logger.info(f"Converting {image_path.suffix} to supported format: {image_path}")
                return self._convert_image_to_supported_format(image_path)
            
            else:
                logger.warning(f"Unsupported image format: {image_path.suffix} for file {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def retry_with_exponential_backoff(max_retries=5, base_delay=10, max_delay=90):
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
                
                return None
            return wrapper
        return decorator

    @retry_with_exponential_backoff(max_retries=5, base_delay=10, max_delay=90)
    def generate_embedding_from_path(self, image_path: Union[str, Path], 
                                   contextual_text: str = "") -> Optional[np.ndarray]:
        """
        Generate embedding for a single image from file path.
        
        Args:
            image_path: Path to image file
            contextual_text: Optional contextual text for the image
            
        Returns:
            Image embedding as numpy array or None if failed
        """
        try:
            # Load and process image
            image_bytes = self._load_image_file(image_path)
            if image_bytes is None:
                return None
            
            image = Image.load_from_file(str(image_path))
            
            # Generate embedding using the model
            embeddings = self.model.get_embeddings(
                image=image,
                contextual_text=contextual_text, # title
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
        """
        Process a batch of images and generate embeddings.
        
        Args:
            image_paths: List of paths to image files
            contextual_texts: Optional list of contextual texts (same length as image_paths)
            
        Returns:
            List of tuples (image_path, embedding) where embedding can be None if failed
        """
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
        """
        LangChain interface method for embedding multiple documents.
        For images, 'texts' should be image file paths.
        
        Args:
            texts: List of image file paths
            
        Returns:
            List of embeddings as lists of floats
        """
        embeddings = []
        
        for image_path in texts:
            contextual_text = self._get_contextual_text_from_path(image_path)
            
            embedding = self.generate_embedding_from_path(image_path, contextual_text=contextual_text)
            if embedding is not None:
                embeddings.append(embedding.tolist())
            else:
                embeddings.append([0.0] * self.DIMENSION)
                logger.warning(f"Failed to embed image {image_path}, using zero vector")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        LangChain interface method for embedding a single query.
        Supports both text queries and image path queries.
        
        Args:
            text: Either a text query or path to an image file
            
        Returns:
            Embedding as list of floats
        """
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
        """
        Helper method to determine if input text is likely an image file path.
        """
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
        """
        Make the class callable for FAISS compatibility.
        This is a wrapper around embed_query.
        
        Args:
            text: Query text or image path
            
        Returns:
            Embedding as list of floats
        """
        return self.embed_query(text)
    
    def scan_directory(self, directory_path: Union[str, Path], 
                      recursive: bool = True) -> List[Path]:
        """
        Scan directory for supported image files.
        
        Args:
            directory_path: Path to directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            List of paths to supported image files
        """
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
    """
    Process images from a specific folder structure with metadata matching.
    
    Expected structure:
    - metadata.json (contains article info with titles)
    - img/ folder (contains numbered image files)
    """
    
    PROJECT_ID = os.getenv("PROJECT_ID")
    if not PROJECT_ID:
        print("Error: GOOGLE_CLOUD_PROJECT not found in environment")
        return

    service = ImageEmbeddingService(PROJECT_ID)
   
    metadata_file = "metadata.json"  
    img_folder = "img"         

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} articles")
        
        img_folder_path = Path(img_folder)
        if not img_folder_path.exists():
            print(f"Image folder not found: {img_folder}")
            return
        
        supported_images = service.scan_directory(img_folder_path, recursive=False)
        print(f"Found {len(supported_images)} images")

        matched_images = []
        
        for img_path in supported_images:
            img_filename = img_path.name
            try:
                img_id = img_filename.split('_')[0]
                if img_id in metadata:
                    matched_images.append({
                        'path': img_path,
                        'id': img_id,
                        'title': metadata[img_id]['title'],
                        'filename': img_filename
                    })
            except (IndexError, ValueError):
                continue
        
        print(f"Matched {len(matched_images)} images to metadata")
        
        if not matched_images:
            print("No images matched to metadata")
            return
        
        print(f"Processing {len(matched_images)} images...")
        
        image_paths = [item['path'] for item in matched_images]
        contextual_texts = [item['title'] for item in matched_images]
        
        batch_results = service.process_image_batch(image_paths, contextual_texts)
        
        all_results = []
        successful = 0
        
        for i, (path, embedding) in enumerate(batch_results):
            item = matched_images[i]
            success = embedding is not None
            if success:
                successful += 1
            
            all_results.append({
                'id': item['id'],
                'title': item['title'],
                'filename': item['filename'],
                'path': str(path),
                'embedding': embedding,
                'success': success
            })
        
        print(f"Results: {successful}/{len(all_results)} successful ({successful/len(all_results)*100:.0f}%)")
        
        # Save results
        results_file = "image_embeddings_results.json"
        json_results = []
        
        for result in all_results:
            json_result = result.copy()
            if result['embedding'] is not None:
                json_result['embedding'] = result['embedding'].tolist()
                json_result['embedding_shape'] = result['embedding'].shape
            else:
                json_result['embedding'] = None
                json_result['embedding_shape'] = None
            json_results.append(json_result)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {results_file}")
        
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_file}")
    except json.JSONDecodeError as e:
        print(f"Error parsing metadata: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()