# Create this as image_processor_enhanced.py or update existing image_processor.py

import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from config import PipelineConfig
from logger import PipelineLogger
from filter_factory import FilterFactory
from hair_removal import HairRemovalProcessor
from performance_monitor import PerformanceMonitor

class StructurePreservingImageProcessor:
    """Image processor that preserves train/test and benign/malignant structure"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger.setup_logger(
            "StructurePreservingImageProcessor", 
            config.logging_level
        )
        
        self.hair_processor = HairRemovalProcessor(
            config.processing, 
            self.logger
        )
        
        # Setup GPU if available
        if config.processing.enable_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cv2.cuda.setDevice(0)
            cv2.setUseOptimized(True)
            self.logger.info("🚀 GPU acceleration enabled")
        else:
            self.logger.info("💻 Using CPU processing")
    
    def _apply_medical_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply medical image specific enhancements"""
        try:
            # Convert to LAB color space for better skin tone handling
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge back and convert to BGR
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Medical enhancement failed: {str(e)}, using original")
            return image
    
    def process_single_image(self, image_info: Dict) -> bool:
        """Process a single image while preserving structure"""
        try:
            image_path = image_info['path']
            relative_path = image_info['relative_path']
            filter_name = image_info['filter_name']
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Apply medical enhancement
            enhanced_image = self._apply_medical_enhancement(image)
            
            # Detect and remove hair
            processed_image, has_hair = self.hair_processor.detect_and_remove_hair(enhanced_image)
            
            # Apply filter
            filter_instance = FilterFactory.create_filter(filter_name, self.config.filters, self.logger)
            filtered_image = filter_instance.apply(processed_image)
            
            # Resize to target size
            resized_image = cv2.resize(filtered_image, self.config.processing.target_size)
            
            # Create output path preserving structure
            output_dir = Path(self.config.processing.output_path) / filter_name / Path(relative_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            base_name = Path(image_path).stem
            hair_prefix = "with_hair" if has_hair else "without_hair"
            output_filename = f"{hair_prefix}_{base_name}_{filter_name}.jpg"
            output_path = output_dir / output_filename
            
            # Save processed image
            cv2.imwrite(str(output_path), resized_image)
            
            self.logger.debug(f"Processed: {relative_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {image_info.get('path', 'unknown')}: {str(e)}")
            return False
    
    def _get_structured_image_paths(self) -> Dict[str, List[Dict]]:
        """Get all image paths organized by dataset structure"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        structured_paths = {}
        
        input_path = Path(self.config.processing.input_path)
        
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, input_path)
                    
                    path_parts = Path(relative_path).parts
                    if len(path_parts) >= 2:
                        category = os.path.join(path_parts[0], path_parts[1])
                        
                        if category not in structured_paths:
                            structured_paths[category] = []
                        
                        structured_paths[category].append({
                            'path': full_path,
                            'relative_path': relative_path,
                            'category': category
                        })
        
        total_images = sum(len(paths) for paths in structured_paths.values())
        self.logger.info(f"📊 Found {total_images} images in {len(structured_paths)} categories:")
        for category, paths in structured_paths.items():
            self.logger.info(f"   {category}: {len(paths)} images")
        
        return structured_paths
    
    def process_dataset_structured(self, filter_names: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
        """Process entire dataset with structure preservation"""
        if filter_names is None:
            filter_names = FilterFactory.get_available_filters()
        
        structured_paths = self._get_structured_image_paths()
        total_images = sum(len(paths) for paths in structured_paths.values())
        
        self.logger.info(f"🔄 Processing {total_images} images with filters: {filter_names}")
        
        results = {}
        start_time = time.time()
        
        for filter_name in filter_names:
            filter_start = time.time()
            self.logger.info(f"🎯 Starting filter: {filter_name}")
            
            filter_results = {}
            filter_total_successful = 0
            
            for category, image_infos in structured_paths.items():
                category_start = time.time()
                
                # Add filter name to each image info
                for info in image_infos:
                    info['filter_name'] = filter_name
                
                # Process in batches
                batch_results = []
                batch_size = self.config.processing.batch_size
                
                for i in range(0, len(image_infos), batch_size):
                    batch_infos = image_infos[i:i + batch_size]
                    
                    with ThreadPoolExecutor(max_workers=self.config.processing.num_workers) as executor:
                        futures = [executor.submit(self.process_single_image, info) for info in batch_infos]
                        batch_result = [future.result() for future in futures]
                    
                    batch_results.extend(batch_result)
                
                successful = sum(batch_results)
                filter_results[category] = successful
                filter_total_successful += successful
                
                category_time = time.time() - category_start
                self.logger.info(f"   ✅ {category}: {successful}/{len(image_infos)} images in {category_time:.1f}s")
            
            results[filter_name] = filter_results
            filter_time = time.time() - filter_start
            self.logger.info(f"🎉 Filter {filter_name} completed: {filter_total_successful}/{total_images} images in {filter_time:.1f}s")
        
        total_time = time.time() - start_time
        self.logger.info(f"⚡ Total processing time: {total_time:.1f}s")
        
        return results

# For backward compatibility, create an alias
ImageProcessor = StructurePreservingImageProcessor