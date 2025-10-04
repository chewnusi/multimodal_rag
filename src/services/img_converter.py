import logging
from typing import Dict, Any, Optional
from pathlib import Path
import io
from PIL import Image

logger = logging.getLogger(__name__)

class ImageConverter:

    @staticmethod
    def convert_to_supported_format(image_path: str, max_size_mb: float = 15.0) -> Optional[Dict[str, Any]]:
       
        try:
            working_path = Path(image_path)
            
            if not working_path.exists():
                logger.warning(f"Image not found at path: {working_path}")

                fallback_path = Path("data/img") / working_path.name
                if fallback_path.exists():
                    working_path = fallback_path
                else:
                    logger.warning(f"Image not found in data/img either: {fallback_path}")
                    return None
            
            img = Image.open(working_path)
            original_format = img.format
            
            target_format = None
            mime_type = None
            
            if original_format in ['JPEG', 'PNG', 'WEBP']:
                with open(working_path, 'rb') as f:
                    original_bytes = f.read()
                    size_mb = len(original_bytes) / (1024 * 1024)
                    
                    if size_mb <= max_size_mb:
                        if original_format == 'JPEG':
                            mime_type = 'image/jpeg'
                        elif original_format == 'PNG':
                            mime_type = 'image/png'
                        elif original_format == 'WEBP':
                            mime_type = 'image/webp'
                        
                        return {
                            'data': original_bytes,
                            'mime_type': mime_type,
                            'format': original_format
                        }
            
            if original_format == 'GIF':
                output = io.BytesIO()
                
                is_animated = getattr(img, 'is_animated', False)
                
                if is_animated:
                    img.save(output, format='WEBP', save_all=True, duration=img.info.get('duration', 100), loop=0)
                else:
                    img.save(output, format='WEBP', quality=85)
                
                target_format = 'WEBP'
                mime_type = 'image/webp'
            
            else:
                output = io.BytesIO()
                
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(output, format='JPEG', quality=85, optimize=True)
                target_format = 'JPEG'
                mime_type = 'image/jpeg'
            
            converted_data = output.getvalue()
            size_mb = len(converted_data) / (1024 * 1024)
            
            if size_mb > max_size_mb:
                logger.warning(f"Converted image exceeds size limit: {working_path} ({size_mb:.2f} MB > {max_size_mb} MB)")
                
                if target_format == 'JPEG':
                    output = io.BytesIO()
                    img.save(output, format='JPEG', quality=70, optimize=True)
                    converted_data = output.getvalue()
                    size_mb = len(converted_data) / (1024 * 1024)
                    
                    if size_mb > max_size_mb:
                        logger.warning(f"Image too large after compression: {size_mb:.2f} MB")
                        return None
            
            return {
                'data': converted_data,
                'mime_type': mime_type,
                'format': target_format
            }
            
        except Exception as e:
            logger.error(f"Error converting image {image_path}: {e}")
            return None
