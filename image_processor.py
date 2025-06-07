# image_processor.py - Structure Preserving
import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict
from pathlib import Path

from config import PipelineConfig
from logger import PipelineLogger
from filter_factory import FilterFactory
from hair_removal import HairRemovalProcessor
from performance_monitor import PerformanceMonitor

class ImageProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger.setup_logger('ImageProcessor', config.logging_level)
        self.hair_processor = HairRemovalProcessor(config.processing, self.logger)
        
        if config.processing.enable_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cv2.cuda.setDevice(0)
            cv2.setUseOptimized(True)
            self.logger.info('🚀 GPU acceleration enabled')
        else:
            self.logger.info('💻 Using CPU processing')
    
    def process_single_image(self, image_info: Dict) -> bool:
        try:
            image_path = image_info['path']
            relative_path = image_info['relative_path']
            filter_name = image_info['filter_name']
            
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f'Failed to load: {image_path}')
                return False
            
            # Gentle enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            l_final = cv2.addWeighted(l, 0.8, l_enhanced, 0.2, 0)
            lab_enhanced = cv2.merge([l_final, a, b])
            enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # Hair removal
            processed_image, has_hair = self.hair_processor.detect_and_remove_hair(enhanced_image)
            
            # Apply filter
            filter_instance = FilterFactory.create_filter(filter_name, self.config.filters, self.logger)
            filtered_image = filter_instance.apply(processed_image)
            
            # Resize
            resized_image = cv2.resize(filtered_image, self.config.processing.target_size)
            
            # STRUCTURE PRESERVING OUTPUT PATH
            output_dir = Path(self.config.processing.output_path) / filter_name / Path(relative_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(image_path).stem
            hair_prefix = 'with_hair' if has_hair else 'without_hair'
            output_filename = f'{hair_prefix}_{base_name}_{filter_name}.jpg'
            output_path = output_dir / output_filename
            
            cv2.imwrite(str(output_path), resized_image)
            self.logger.debug(f'✅ {relative_path} -> {filter_name}/{Path(relative_path).parent}/{output_filename}')
            return True
            
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            return False
    
    def _get_structured_image_paths(self) -> Dict[str, List[Dict]]:
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
        self.logger.info(f'📊 Dataset Structure: {total_images} total images')
        for category, paths in structured_paths.items():
            self.logger.info(f'   {category}: {len(paths)} images')
        
        return structured_paths
    
    def process_dataset(self, filter_names: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
        if filter_names is None:
            filter_names = FilterFactory.get_recommended_filters()
        
        structured_paths = self._get_structured_image_paths()
        total_images = sum(len(paths) for paths in structured_paths.values())
        
        if total_images == 0:
            self.logger.error('❌ No images found!')
            return {}
        
        self.logger.info(f'🔄 Processing {total_images} images with {filter_names}')
        
        results = {}
        start_time = time.time()
        
        for filter_name in filter_names:
            self.logger.info(f'🎯 Starting filter: {filter_name}')
            filter_results = {}
            filter_total = 0
            
            for category, image_infos in structured_paths.items():
                # Add filter name
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
                filter_total += successful
                self.logger.info(f'   ✅ {category}: {successful}/{len(image_infos)} images')
            
            results[filter_name] = filter_results
            self.logger.info(f'🎉 {filter_name}: {filter_total}/{total_images} total')
        
        total_time = time.time() - start_time
        self.logger.info(f'⚡ Total time: {total_time:.1f}s')
        return results
