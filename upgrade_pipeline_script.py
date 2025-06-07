#!/usr/bin/env python3
"""
Pipeline Upgrade Script - Enhanced Melanoma Detection
Upgrades the existing pipeline with:
1. Structure-preserving processing (train/test, benign/malignant)
2. Enhanced medical filters for melanoma detection
3. Better performance and medical image optimizations
"""

import os
from pathlib import Path

def upgrade_pipeline():
    """Upgrade the pipeline with enhanced features"""
    
    print("🏥 Upgrading Melanoma Detection Pipeline")
    print("=" * 60)
    
    # Create enhanced medical filters
    medical_bilateral = '''import cv2
import numpy as np
from base_filter import BaseFilter

class MedicalBilateralFilter(BaseFilter):
    """Bilateral filter optimized for dermoscopic skin images"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply medical-optimized bilateral filter"""
        if not self.validate_image(image):
            return image
            
        try:
            # Parameters specifically tuned for skin lesion images
            d = 15  # Larger diameter for skin texture
            sigma_color = 80  # Higher for skin tone variations
            sigma_space = 80  # Better edge preservation
            
            # Apply bilateral filter
            filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
            # Additional melanoma-specific enhancement
            enhanced = self._enhance_lesion_contrast(filtered)
            
            self.logger.debug(f"Applied medical bilateral filter with d={d}")
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error applying medical bilateral filter: {str(e)}")
            return image
    
    def _enhance_lesion_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast specifically for lesion detection"""
        try:
            # Convert to LAB for better skin processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply gentle CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Enhance A and B channels for better color lesion detection
            a_enhanced = cv2.addWeighted(a, 1.2, np.zeros_like(a), 0, 5)
            b_enhanced = cv2.addWeighted(b, 1.1, np.zeros_like(b), 0, 3)
            
            # Merge and convert back
            lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception:
            return image
    
    def get_name(self) -> str:
        return "medical_bilateral"
'''

    lesion_enhancement = '''import cv2
import numpy as np
from base_filter import BaseFilter

class LesionEnhancementFilter(BaseFilter):
    """Specialized filter for melanoma lesion enhancement"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply lesion-specific enhancement"""
        if not self.validate_image(image):
            return image
            
        try:
            # Multi-step enhancement for melanoma detection
            enhanced = self._apply_abcd_enhancement(image)
            enhanced = self._enhance_border_detection(enhanced)
            enhanced = self._improve_color_contrast(enhanced)
            
            self.logger.debug("Applied lesion enhancement filter")
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error applying lesion enhancement: {str(e)}")
            return image
    
    def _apply_abcd_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance features relevant to ABCD criteria for melanoma"""
        try:
            # Convert to different color spaces for different criteria
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            h, s, v = cv2.split(hsv)
            l, a, b_lab = cv2.split(lab)
            
            # Asymmetry enhancement (edge detection)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Border enhancement for irregular borders
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            border_enhanced = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Color enhancement for melanoma-specific colors
            s_enhanced = cv2.addWeighted(s, 1.3, np.zeros_like(s), 0, 10)
            v_enhanced = cv2.addWeighted(v, 1.1, border_enhanced // 5, 0.2, 0)
            
            # Merge back
            hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
            enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
            
            return enhanced
            
        except Exception:
            return image
    
    def _enhance_border_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance border detection for irregular melanoma borders"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale edge detection
            edges1 = cv2.Canny(gray, 30, 100)
            edges2 = cv2.Canny(gray, 50, 150)
            edges3 = cv2.Canny(gray, 80, 200)
            
            # Combine edges
            combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
            
            # Apply gentle morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            refined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
            
            # Blend with original image
            enhanced = image.copy()
            for i in range(3):
                enhanced[:, :, i] = cv2.addWeighted(
                    image[:, :, i], 0.85, 
                    refined_edges, 0.15, 0
                )
            
            return enhanced
            
        except Exception:
            return image
    
    def _improve_color_contrast(self, image: np.ndarray) -> np.ndarray:
        """Improve color contrast for melanoma color variations"""
        try:
            # Convert to LAB for better color processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Enhance L channel with adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            l_enhanced = clahe.apply(l)
            
            # Enhance color channels for melanoma detection
            a_enhanced = cv2.addWeighted(a, 1.25, np.zeros_like(a), 0, 8)
            b_enhanced = cv2.addWeighted(b, 1.15, np.zeros_like(b), 0, 5)
            
            # Merge and convert back
            lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception:
            return image
    
    def get_name(self) -> str:
        return "lesion_enhancement"
'''

    optimized_wavelet = '''import cv2
import numpy as np
import pywt
from typing import Tuple
from base_filter import BaseFilter

class OptimizedWaveletFilter(BaseFilter):
    """Wavelet filter optimized for medical image denoising"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply optimized wavelet denoising"""
        if not self.validate_image(image):
            return image
            
        try:
            # Use different processing for color vs grayscale
            if len(image.shape) == 3:
                # Process in YUV color space for better medical image handling
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                y, u, v = cv2.split(yuv)
                
                # Apply wavelet denoising to Y channel (luminance)
                y_denoised = self._apply_medical_wavelet_denoising(y)
                
                # Light denoising on color channels
                u_denoised = self._apply_light_wavelet_denoising(u)
                v_denoised = self._apply_light_wavelet_denoising(v)
                
                # Merge and convert back
                yuv_denoised = cv2.merge([y_denoised, u_denoised, v_denoised])
                filtered_image = cv2.cvtColor(yuv_denoised, cv2.COLOR_YUV2BGR)
            else:
                filtered_image = self._apply_medical_wavelet_denoising(image)
            
            self.logger.debug(f"Applied optimized wavelet filter")
            return filtered_image
            
        except Exception as e:
            self.logger.error(f"Error applying optimized wavelet filter: {str(e)}")
            return image
    
    def _apply_medical_wavelet_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising optimized for medical images"""
        try:
            # Convert to float
            img_float = image.astype(np.float32) / 255.0
            
            # Use Daubechies wavelet for medical images (better than Haar)
            wavelet = 'db4'  # Better for medical image textures
            levels = 3  # More levels for better denoising
            
            # Decompose
            coeffs = pywt.wavedec2(img_float, wavelet, level=levels)
            
            # Estimate noise more accurately for medical images
            sigma = self._estimate_medical_noise(coeffs[-1])
            
            # Use BayesShrink method for medical images
            threshold = sigma * np.sqrt(2 * np.log(img_float.size))
            
            # Apply soft thresholding with medical-optimized parameters
            coeffs_thresh = [coeffs[0]]  # Keep approximation
            
            for i in range(1, len(coeffs)):
                detail_coeffs = coeffs[i]
                # Adaptive thresholding for each detail level
                level_threshold = threshold / (2 ** (i-1))  # Adaptive threshold
                
                thresholded_details = tuple(
                    pywt.threshold(detail, level_threshold, mode='soft') 
                    for detail in detail_coeffs
                )
                coeffs_thresh.append(thresholded_details)
            
            # Reconstruct
            img_reconstructed = pywt.waverec2(coeffs_thresh, wavelet)
            
            # Convert back to uint8
            img_reconstructed = np.clip(img_reconstructed * 255, 0, 255).astype(np.uint8)
            
            return img_reconstructed
            
        except Exception as e:
            self.logger.error(f"Error in medical wavelet denoising: {str(e)}")
            return image
    
    def _apply_light_wavelet_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply light wavelet denoising for color channels"""
        try:
            img_float = image.astype(np.float32) / 255.0
            
            # Use only 2 levels for color channels
            coeffs = pywt.wavedec2(img_float, 'db2', level=2)
            
            # Light denoising for color preservation
            sigma = self._estimate_medical_noise(coeffs[-1])
            threshold = sigma * 0.5  # Lighter threshold for color channels
            
            coeffs_thresh = [coeffs[0]]
            for i in range(1, len(coeffs)):
                detail_coeffs = coeffs[i]
                thresholded_details = tuple(
                    pywt.threshold(detail, threshold, mode='soft') 
                    for detail in detail_coeffs
                )
                coeffs_thresh.append(thresholded_details)
            
            img_reconstructed = pywt.waverec2(coeffs_thresh, 'db2')
            return np.clip(img_reconstructed * 255, 0, 255).astype(np.uint8)
            
        except Exception:
            return image
    
    def _estimate_medical_noise(self, detail_coeffs: Tuple) -> float:
        """Improved noise estimation for medical images"""
        try:
            # Use the diagonal detail coefficients (cD)
            diagonal_details = detail_coeffs[2]
            
            # Robust noise estimation using MAD
            mad = np.median(np.abs(diagonal_details - np.median(diagonal_details)))
            sigma = mad / 0.6745
            
            # Clamp to reasonable values for medical images
            sigma = np.clip(sigma, 0.01, 0.3)
            
            return sigma
            
        except Exception:
            return 0.05  # Safe fallback
    
    def get_name(self) -> str:
        return "optimized_wavelet"
'''

    adaptive_gaussian = '''import cv2
import numpy as np
from base_filter import BaseFilter

class AdaptiveGaussianFilter(BaseFilter):
    """Adaptive Gaussian filter based on local image properties"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive Gaussian filtering"""
        if not self.validate_image(image):
            return image
            
        try:
            # Convert to grayscale for edge detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detect edges to adapt filtering
            edges = cv2.Canny(gray, 50, 150)
            
            # Create edge mask
            edge_mask = edges > 0
            
            # Apply different filtering based on edge presence
            filtered = np.zeros_like(image)
            
            # Strong smoothing in non-edge areas
            smooth_strong = cv2.GaussianBlur(image, (9, 9), 2.0)
            
            # Light smoothing in edge areas
            smooth_light = cv2.GaussianBlur(image, (3, 3), 0.8)
            
            # Combine based on edge mask
            if len(image.shape) == 3:
                for i in range(3):
                    filtered[:, :, i] = np.where(
                        edge_mask, 
                        smooth_light[:, :, i], 
                        smooth_strong[:, :, i]
                    )
            else:
                filtered = np.where(edge_mask, smooth_light, smooth_strong)
            
            self.logger.debug("Applied adaptive Gaussian filter")
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error applying adaptive Gaussian filter: {str(e)}")
            return image
    
    def get_name(self) -> str:
        return "adaptive_gaussian"
'''

    # Updated image processor with structure preservation
    updated_image_processor = '''# Updated image_processor.py with structure preservation
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
'''

    # Updated filter factory
    updated_filter_factory = '''import logging
from typing import Dict, Type, List
from base_filter import BaseFilter
from config import FilterConfig

# Import all filter implementations
from filters.average_blur_filter import AverageBlurFilter
from filters.gaussian_filter import GaussianFilter
from filters.fourier_filter import FourierFilter
from filters.wavelet_filter import WaveletFilter
from filters.bilateral_filter import BilateralFilter

# Import enhanced medical filters
from filters.medical_bilateral_filter import MedicalBilateralFilter
from filters.lesion_enhancement_filter import LesionEnhancementFilter
from filters.optimized_wavelet_filter import OptimizedWaveletFilter
from filters.adaptive_gaussian_filter import AdaptiveGaussianFilter

class FilterFactory:
    """Factory for creating filter instances"""
    
    _filters: Dict[str, Type[BaseFilter]] = {
        # Original filters
        'average_blur': AverageBlurFilter,
        'gaussian': GaussianFilter,
        'fourier': FourierFilter,
        'wavelet': WaveletFilter,
        'bilateral': BilateralFilter,
        
        # Enhanced medical filters
        'medical_bilateral': MedicalBilateralFilter,
        'lesion_enhancement': LesionEnhancementFilter,
        'optimized_wavelet': OptimizedWaveletFilter,
        'adaptive_gaussian': AdaptiveGaussianFilter,
    }
    
    @classmethod
    def create_filter(cls, filter_name: str, config: FilterConfig, 
                     logger: logging.Logger) -> BaseFilter:
        """Create a filter instance by name"""
        if filter_name not in cls._filters:
            raise ValueError(f"Unknown filter: {filter_name}. "
                           f"Available filters: {list(cls._filters.keys())}")
        
        filter_class = cls._filters[filter_name]
        return filter_class(config, logger)
    
    @classmethod
    def get_available_filters(cls) -> List[str]:
        """Get list of available filter names"""
        return list(cls._filters.keys())
    
    @classmethod
    def get_medical_filters(cls) -> List[str]:
        """Get list of medical-optimized filter names"""
        return ['medical_bilateral', 'lesion_enhancement', 'optimized_wavelet', 'adaptive_gaussian']
    
    @classmethod
    def get_basic_filters(cls) -> List[str]:
        """Get list of basic filter names"""
        return ['average_blur', 'gaussian', 'fourier', 'wavelet', 'bilateral']
'''

    # Updated main.py
    updated_main = '''# Updated main.py with enhanced pipeline
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from logger import PipelineLogger
from performance_monitor import PerformanceMonitor

# Try to import the enhanced processor, fallback to original
try:
    from image_processor import StructurePreservingImageProcessor as ImageProcessor
except ImportError:
    from image_processor import ImageProcessor

from filter_factory import FilterFactory

def main():
    """Enhanced main with structure preservation and medical filters"""
    print("🏥 Enhanced Melanoma Detection Pipeline")
    print("=" * 60)
    
    try:
        # Load configuration
        config = ConfigManager.load_config()
        
        # Setup logger
        logger = PipelineLogger.setup_logger("Main", config.logging_level)
        
        # Initialize performance monitor
        monitor = PerformanceMonitor(logger)
        
        # Initialize processor
        processor = ImageProcessor(config)
        
        # Log startup information
        logger.info("🚀 Starting enhanced melanoma processing pipeline")
        logger.info(f"📁 Input path: {config.processing.input_path}")
        logger.info(f"📁 Output path: {config.processing.output_path}")
        logger.info(f"🎯 Available filters: {FilterFactory.get_available_filters()}")
        logger.info(f"🏥 Medical filters: {FilterFactory.get_medical_filters()}")
        
        # Check input directory
        if not os.path.exists(config.processing.input_path):
            logger.error(f"❌ Input directory not found: {config.processing.input_path}")
            return
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Process with medical filters for better melanoma detection
        medical_filters = FilterFactory.get_medical_filters()
        print(f"\\n🔄 Processing with enhanced medical filters: {medical_filters}")
        
        # Check if we have the enhanced processor
        if hasattr(processor, 'process_dataset_structured'):
            results = processor.process_dataset_structured(medical_filters)
        else:
            results = processor.process_dataset(medical_filters)
        
        # Stop monitoring
        stats = monitor.stop_monitoring()
        
        # Display results
        print("\\n📊 Processing Results:")
        print("=" * 40)
        
        if isinstance(results, dict) and any(isinstance(v, dict) for v in results.values()):
            # Structure-preserving results
            for filter_name, filter_results in results.items():
                print(f"\\n🎯 {filter_name.upper()}:")
                if isinstance(filter_results, dict):
                    total_for_filter = 0
                    for category, count in filter_results.items():
                        print(f"   {category:20s}: {count:4d} images")
                        total_for_filter += count
                    print(f"   {'TOTAL':20s}: {total_for_filter:4d} images")
                else:
                    print(f"   Total: {filter_results} images")
        else:
            # Basic results
            for filter_name, count in results.items():
                print(f"  {filter_name:20s}: {count:4d} images")
        
        print(f"\\n⚡ Performance: {stats}")
        print(f"\\n✅ Enhanced pipeline completed!")
        print(f"📁 Results in: {config.processing.output_path}")
        
        # Show expected structure
        print(f"\\n📁 Output Structure:")
        print(f"   processed_output/")
        print(f"   ├── medical_bilateral/")
        print(f"   │   ├── train/benign/")
        print(f"   │   ├── train/malignant/")
        print(f"   │   ├── test/benign/")
        print(f"   │   └── test/malignant/")
        print(f"   ├── lesion_enhancement/")
        print(f"   │   └── (same structure...)")
        print(f"   └── ... (other filters)")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
'''
    
    # Write all new filter files
    files_to_create = {
        'filters/medical_bilateral_filter.py': medical_bilateral,
        'filters/lesion_enhancement_filter.py': lesion_enhancement,
        'filters/optimized_wavelet_filter.py': optimized_wavelet,
        'filters/adaptive_gaussian_filter.py': adaptive_gaussian,
    }
    
    # Write enhanced files
    for file_path, content in files_to_create.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✅ Created: {file_path}")
    
    # Update existing files
    with open('filter_factory.py', 'w') as f:
        f.write(updated_filter_factory)
    print(f"✅ Updated: filter_factory.py")
    
    # Create enhanced image processor
    with open('image_processor_enhanced.py', 'w') as f:
        f.write(updated_image_processor)
    print(f"✅ Created: image_processor_enhanced.py")
    
    # Create enhanced main
    with open('main_enhanced.py', 'w') as f:
        f.write(updated_main)
    print(f"✅ Created: main_enhanced.py")
    
    # Update filters __init__.py
    filters_init_updated = '''"""
Enhanced Filters package for melanoma detection pipeline
"""

# Original filters
from .average_blur_filter import AverageBlurFilter
from .gaussian_filter import GaussianFilter
from .fourier_filter import FourierFilter
from .wavelet_filter import WaveletFilter
from .bilateral_filter import BilateralFilter

# Enhanced medical filters
from .medical_bilateral_filter import MedicalBilateralFilter
from .lesion_enhancement_filter import LesionEnhancementFilter
from .optimized_wavelet_filter import OptimizedWaveletFilter
from .adaptive_gaussian_filter import AdaptiveGaussianFilter

__all__ = [
    # Original filters
    'AverageBlurFilter',
    'GaussianFilter', 
    'FourierFilter',
    'WaveletFilter',
    'BilateralFilter',
    
    # Enhanced medical filters
    'MedicalBilateralFilter',
    'LesionEnhancementFilter',
    'OptimizedWaveletFilter',
    'AdaptiveGaussianFilter',
]
'''
    
    with open('filters/__init__.py', 'w') as f:
        f.write(filters_init_updated)
    print(f"✅ Updated: filters/__init__.py")
    
    print("\n🎉 Pipeline Upgrade Completed!")
    print("\n📋 What's New:")
    print("✅ Structure-preserving processing (train/test, benign/malignant)")
    print("✅ Medical Bilateral Filter - Optimized for skin lesions")
    print("✅ Lesion Enhancement Filter - ABCD criteria enhancement")
    print("✅ Optimized Wavelet Filter - Better medical image denoising")
    print("✅ Adaptive Gaussian Filter - Edge-aware smoothing")
    print("✅ CLAHE preprocessing for better contrast")
    print("✅ LAB color space processing for skin tones")
    
    print("\n🚀 Next Steps:")
    print("1. Run enhanced pipeline: python main_enhanced.py")
    print("2. Or use medical filters only: python main.py (with updated filter_factory)")
    
    print("\n📊 Expected Output Structure:")
    print("processed_output/")
    print("├── medical_bilateral/")
    print("│   ├── train/benign/ & train/malignant/")
    print("│   └── test/benign/ & test/malignant/")
    print("├── lesion_enhancement/")
    print("├── optimized_wavelet/")
    print("└── adaptive_gaussian/")

if __name__ == "__main__":
    upgrade_pipeline()