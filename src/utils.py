import json
import logging
import time
import random
from typing import Optional, Dict, Any
from pathlib import Path
from functools import wraps

from langchain.schema import Document

logger = logging.getLogger(__name__)


def load_metadata(metadata_file: Path) -> Dict[str, Any]:
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.debug(f"Loaded metadata for {len(metadata)} articles from {metadata_file}")
        return metadata
    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {metadata_file}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metadata from {metadata_file}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading metadata from {metadata_file}: {e}")
        return {}


def extract_article_id_from_filename(filename: str) -> Optional[str]:
    try:
        article_id = filename.split('_')[0]
        
        if article_id and article_id.isdigit():
            return article_id
        else:
            logger.warning(f"Invalid article ID format in filename: {filename}")
            return None
            
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not extract article ID from filename '{filename}': {e}")
        return None


def create_text_document(article_id: str, content: str, filename: str, metadata: Dict[str, Any]) -> Optional[Document]:
    if article_id not in metadata:
        logger.warning(f"No metadata found for article ID: {article_id}")
        return None
    
    article_meta = metadata[article_id]
    
    document = Document(
        page_content=content,
        metadata={
            "article_id": article_id,
            "title": article_meta.get("title", ""),
            "url": article_meta.get("url", ""),
            "date": article_meta.get("date", ""),
            "filename": filename,
            "content_type": "text"
        }
    )
    
    return document


def create_image_document(article_id: str, image_path: Path, metadata: Dict[str, Any]) -> Optional[Document]:
    if article_id not in metadata:
        logger.warning(f"No metadata found for article ID: {article_id}")
        return None
    
    article_meta = metadata[article_id]
    
    document = Document(
        page_content=str(image_path),
        metadata={
            "article_id": article_id,
            "title": article_meta.get("title", ""),
            "url": article_meta.get("url", ""),
            "date": article_meta.get("date", ""),
            "filename": image_path.name,
            "content_type": "image"
        }
    )
    
    return document


def retry_with_exponential_backoff(max_retries: int = 3, base_delay: float = 5, max_delay: float = 60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    is_quota_error = any(keyword in error_msg for keyword in [
                        'quota', '429', 'rate limit', 'exceeded', 'too many requests'
                    ])
                    
                    if is_quota_error and attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        logger.warning(f"Quota/rate limit exceeded, retrying in {delay:.1f}s "
                                     f"(attempt {attempt + 1}/{max_retries + 1})")
                        time.sleep(delay)
                        continue
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for function {func.__name__}")
                    raise e
            
            return None
        return wrapper
    return decorator


def get_contextual_text_from_path(image_path: str, metadata: Dict[str, Any]) -> str:
    try:
        filename = Path(image_path).name
        article_id = extract_article_id_from_filename(filename)
        
        if article_id and article_id in metadata:
            return metadata[article_id].get("title", "")
        else:
            logger.debug(f"No contextual text found for {image_path}")
            return ""
            
    except (IndexError, ValueError, KeyError) as e:
        logger.warning(f"Could not extract contextual text for {image_path}: {e}")
        return ""


def validate_data_structure(data_folder: Path) -> bool:
    required_paths = [
        data_folder / "txt",
        data_folder / "img", 
        data_folder / "metadata.json"
    ]
    
    missing = [path for path in required_paths if not path.exists()]
    
    if missing:
        logger.error("Missing required data folder structure:")
        for path in missing:
            logger.error(f"  - {path}")
        return False
    
    return True


def safe_read_file(file_path: Path) -> Optional[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content if content else None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None