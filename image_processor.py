import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional
import multiprocessing as mp

from config import PipelineConfig
from hair_removal import HairRemovalProcessor
from logger import PipelineLogger
from filter_factory import FilterFactory
import cv2
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

class ImageProcessor:
    """Main image processing pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger.setup_logger(
            "ImageProcessor", 
            config.logging_level
        )
        
        self.hair_processor = HairRemovalProcessor(
            config.processing, 
            self.logger
        )
        
        # Setup GPU if available and enabled
        if config.processing.enable_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cv2.cuda.setDevice(0)
            cv2.setUseOptimized(True)
            self.logger.info("GPU acceleration enabled")
        else:
            self.logger.info("Using CPU processing")
    
    def process_single_image(self, image_path: str, output_dir: str, 
                           filter_name: str) -> bool:
        """Process a single image with specified filter"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Detect and remove hair
            processed_image, has_hair = self.hair_processor.detect_and_remove_hair(image)
            
            # Apply filter
            filter_instance = FilterFactory.create_filter(
                filter_name, 
                self.config.filters, 
                self.logger
            )
            
            filtered_image = filter_instance.apply(processed_image)
            
            # Resize to target size
            resized_image = cv2.resize(
                filtered_image, 
                self.config.processing.target_size
            )
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            hair_prefix = "with_hair" if has_hair else "without_hair"
            output_filename = f"{hair_prefix}_{base_name}_{filter_name}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save processed image
            cv2.imwrite(output_path, resized_image)
            
            self.logger.debug(f"Processed: {image_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return False
    
    def process_batch(self, image_paths: List[str], output_dir: str, 
                     filter_name: str) -> List[bool]:
        """Process a batch of images"""
        results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process with threading for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.config.processing.num_workers) as executor:
            futures = [
                executor.submit(self.process_single_image, path, output_dir, filter_name)
                for path in image_paths
            ]
            
            for future in futures:
                results.append(future.result())
        
        return results
    
    def process_dataset(self, filter_names: Optional[List[str]] = None) -> Dict[str, int]:
        """Process entire dataset with specified filters"""
        if filter_names is None:
            filter_names = FilterFactory.get_available_filters()
        
        # Get all image paths
        image_paths = self._get_image_paths()
        total_images = len(image_paths)
        
        self.logger.info(f"Processing {total_images} images with filters: {filter_names}")
        
        results = {}
        start_time = time.time()
        
        for filter_name in filter_names:
            filter_start = time.time()
            
            # Create filter-specific output directory
            filter_output_dir = os.path.join(
                self.config.processing.output_path, 
                filter_name
            )
            
            # Process in batches
            batch_results = []
            for i in range(0, total_images, self.config.processing.batch_size):
                batch_paths = image_paths[i:i + self.config.processing.batch_size]
                batch_result = self.process_batch(batch_paths, filter_output_dir, filter_name)
                batch_results.extend(batch_result)
            
            successful = sum(batch_results)
            results[filter_name] = successful
            
            filter_time = time.time() - filter_start
            self.logger.info(f"Filter {filter_name}: {successful}/{total_images} "
                           f"images processed in {filter_time:.2f}s")
        
        total_time = time.time() - start_time
        self.logger.info(f"Total processing time: {total_time:.2f}s")
        
        return results
    
    def _get_image_paths(self) -> List[str]:
        """Get all image paths from input directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        input_path = Path(self.config.processing.input_path)
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        self.logger.info(f"Found {len(image_paths)} images")
        return image_paths

