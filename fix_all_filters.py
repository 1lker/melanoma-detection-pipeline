#!/usr/bin/env python3
"""
Quick fix script to create all missing filter files
Run this to solve the ModuleNotFoundError
"""

import os
from pathlib import Path

def create_all_filters():
    """Create all missing filter files"""
    
    print("🔧 Creating all missing filter files...")
    
    # Ensure filters directory exists
    Path("filters").mkdir(exist_ok=True)
    
    # Updated __init__.py for filters
    filters_init = '''"""
Filters package for melanoma detection pipeline
"""

# Import all available filters
try:
    from .average_blur_filter import AverageBlurFilter
except ImportError:
    AverageBlurFilter = None

try:
    from .gaussian_filter import GaussianFilter
except ImportError:
    GaussianFilter = None

try:
    from .fourier_filter import FourierFilter
except ImportError:
    FourierFilter = None

try:
    from .wavelet_filter import WaveletFilter
except ImportError:
    WaveletFilter = None

try:
    from .bilateral_filter import BilateralFilter
except ImportError:
    BilateralFilter = None

try:
    from .medical_bilateral_filter import MedicalBilateralFilter
except ImportError:
    MedicalBilateralFilter = None

try:
    from .lesion_enhancement_filter import LesionEnhancementFilter
except ImportError:
    LesionEnhancementFilter = None

try:
    from .optimized_wavelet_filter import OptimizedWaveletFilter
except ImportError:
    OptimizedWaveletFilter = None

try:
    from .adaptive_gaussian_filter import AdaptiveGaussianFilter
except ImportError:
    AdaptiveGaussianFilter = None

__all__ = [
    'AverageBlurFilter',
    'GaussianFilter', 
    'FourierFilter',
    'WaveletFilter',
    'BilateralFilter',
    'MedicalBilateralFilter',
    'LesionEnhancementFilter',
    'OptimizedWaveletFilter',
    'AdaptiveGaussianFilter',
]
'''

    # Average Blur Filter
    average_blur = '''import cv2
import numpy as np
from base_filter import BaseFilter

class AverageBlurFilter(BaseFilter):
    """Average blur filter implementation"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply average blur filter"""
        if not self.validate_image(image):
            return image
            
        try:
            kernel_size = self.config.kernel_size
            blurred = cv2.blur(image, (kernel_size, kernel_size))
            
            self.logger.debug(f"Applied average blur with kernel size {kernel_size}")
            return blurred
            
        except Exception as e:
            self.logger.error(f"Error applying average blur: {str(e)}")
            return image
    
    def get_name(self) -> str:
        return "average_blur"
'''

    # Gaussian Filter
    gaussian = '''import cv2
import numpy as np
from base_filter import BaseFilter

class GaussianFilter(BaseFilter):
    """Gaussian blur (low-pass) filter implementation"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur filter"""
        if not self.validate_image(image):
            return image
            
        try:
            kernel_size = self.config.kernel_size
            sigma = self.config.gaussian_sigma
            
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # If sigma is 0, calculate it automatically
            if sigma == 0:
                sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
            
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
            self.logger.debug(f"Applied Gaussian blur with kernel size {kernel_size}, sigma {sigma}")
            return blurred
            
        except Exception as e:
            self.logger.error(f"Error applying Gaussian blur: {str(e)}")
            return image
    
    def get_name(self) -> str:
        return "gaussian"
'''

    # Fourier Filter
    fourier = '''import cv2
import numpy as np
from typing import Tuple
from base_filter import BaseFilter

class FourierFilter(BaseFilter):
    """Fourier domain filtering implementation"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Fourier domain filter"""
        if not self.validate_image(image):
            return image
            
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Fourier transform
            dft = np.fft.fft2(gray)
            dft_shifted = np.fft.fftshift(dft)
            
            # Create mask
            mask = self._create_lowpass_mask(gray.shape, self.config.fourier_mask_size)
            
            # Apply mask
            dft_filtered = dft_shifted * mask
            
            # Inverse transform
            dft_ishifted = np.fft.ifftshift(dft_filtered)
            img_filtered = np.fft.ifft2(dft_ishifted)
            img_filtered = np.abs(img_filtered).astype(np.uint8)
            
            # Convert back to original format if needed
            if len(image.shape) == 3:
                img_filtered = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
            
            self.logger.debug(f"Applied Fourier filter with mask size {self.config.fourier_mask_size}")
            return img_filtered
            
        except Exception as e:
            self.logger.error(f"Error applying Fourier filter: {str(e)}")
            return image
    
    def _create_lowpass_mask(self, shape: Tuple[int, int], mask_size: int) -> np.ndarray:
        """Create low-pass mask for Fourier filtering"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.zeros((rows, cols), dtype=np.uint8)
        mask[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 1
        
        return mask
    
    def get_name(self) -> str:
        return "fourier"
'''

    # Fixed Wavelet Filter
    wavelet = '''import cv2
import numpy as np
import pywt
from typing import Tuple
from base_filter import BaseFilter

class WaveletFilter(BaseFilter):
    """Wavelet denoising filter implementation - FIXED VERSION"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising filter"""
        if not self.validate_image(image):
            return image
            
        try:
            # Handle color images by processing each channel
            if len(image.shape) == 3:
                channels = cv2.split(image)
                filtered_channels = []
                
                for channel in channels:
                    filtered_channel = self._apply_wavelet_single_channel(channel)
                    filtered_channels.append(filtered_channel)
                
                filtered_image = cv2.merge(filtered_channels)
            else:
                filtered_image = self._apply_wavelet_single_channel(image)
            
            self.logger.debug(f"Applied wavelet filter with {self.config.wavelet_type} wavelet")
            return filtered_image
            
        except Exception as e:
            self.logger.error(f"Error applying wavelet filter: {str(e)}")
            return image
    
    def _apply_wavelet_single_channel(self, image: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising to a single channel - FIXED VERSION"""
        try:
            # Convert to float
            img_float = image.astype(np.float32) / 255.0
            
            # Decompose using wavedec2
            coeffs = pywt.wavedec2(img_float, self.config.wavelet_type, 
                                  level=self.config.wavelet_levels)
            
            # Estimate noise level from the finest level detail coefficients
            if len(coeffs) > 1:
                # Get the finest level detail coefficients (last tuple)
                finest_details = coeffs[-1]
                # Use the diagonal detail coefficients for noise estimation
                sigma = self._estimate_noise(finest_details[2])  # cD_1 (diagonal)
            else:
                sigma = 0.1  # fallback value
            
            # Calculate threshold
            threshold = sigma * np.sqrt(2 * np.log(img_float.size))
            
            # Apply soft thresholding to detail coefficients
            coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients unchanged
            
            # Process each level of detail coefficients
            for i in range(1, len(coeffs)):
                detail_coeffs = coeffs[i]
                # detail_coeffs is a tuple: (cH, cV, cD)
                thresholded_details = tuple(
                    pywt.threshold(detail, threshold, mode='soft') 
                    for detail in detail_coeffs
                )
                coeffs_thresh.append(thresholded_details)
            
            # Reconstruct
            img_reconstructed = pywt.waverec2(coeffs_thresh, self.config.wavelet_type)
            
            # Convert back to uint8
            img_reconstructed = np.clip(img_reconstructed * 255, 0, 255).astype(np.uint8)
            
            return img_reconstructed
            
        except Exception as e:
            self.logger.error(f"Error in wavelet single channel processing: {str(e)}")
            # Fallback: return original image
            return image
    
    def _estimate_noise(self, detail_coeffs: np.ndarray) -> float:
        """Estimate noise level from detail coefficients"""
        try:
            # Use median absolute deviation (MAD) method
            return np.median(np.abs(detail_coeffs)) / 0.6745
        except:
            return 0.1  # fallback value
    
    def get_name(self) -> str:
        return "wavelet"
'''

    # Bilateral Filter
    bilateral = '''import cv2
import numpy as np
from base_filter import BaseFilter

class BilateralFilter(BaseFilter):
    """Bilateral filter implementation for edge-preserving smoothing"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter"""
        if not self.validate_image(image):
            return image
            
        try:
            d = self.config.bilateral_d
            sigma_color = self.config.bilateral_sigma_color
            sigma_space = self.config.bilateral_sigma_space
            
            filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
            self.logger.debug(f"Applied bilateral filter with d={d}, "
                            f"sigma_color={sigma_color}, sigma_space={sigma_space}")
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error applying bilateral filter: {str(e)}")
            return image
    
    def get_name(self) -> str:
        return "bilateral"
'''

    # Medical Bilateral Filter
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

    # Lesion Enhancement Filter
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
            # Convert to HSV for better color processing
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
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

    # Optimized Wavelet Filter
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

    # Adaptive Gaussian Filter
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

    # Updated Filter Factory
    updated_filter_factory = '''import logging
from typing import Dict, Type, List
from base_filter import BaseFilter
from config import FilterConfig

# Import all filter implementations with error handling
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
            available = list(cls._filters.keys())
            raise ValueError(f"Unknown filter: {filter_name}. Available filters: {available}")
        
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

    # Write all files
    files_to_create = {
        'filters/__init__.py': filters_init,
        'filters/average_blur_filter.py': average_blur,
        'filters/gaussian_filter.py': gaussian,
        'filters/fourier_filter.py': fourier,
        'filters/wavelet_filter.py': wavelet,
        'filters/bilateral_filter.py': bilateral,
        'filters/medical_bilateral_filter.py': medical_bilateral,
        'filters/lesion_enhancement_filter.py': lesion_enhancement,
        'filters/optimized_wavelet_filter.py': optimized_wavelet,
        'filters/adaptive_gaussian_filter.py': adaptive_gaussian,
    }
    
    for file_path, content in files_to_create.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✅ Created: {file_path}")
    
    # Update filter factory
    with open('filter_factory.py', 'w') as f:
        f.write(updated_filter_factory)
    print(f"✅ Updated: filter_factory.py")
    
    print("\n🎉 All filter files created successfully!")
    print("\n📋 Available Filters:")
    print("✅ Basic Filters: average_blur, gaussian, fourier, wavelet, bilateral")
    print("✅ Medical Filters: medical_bilateral, lesion_enhancement, optimized_wavelet, adaptive_gaussian")
    
    print("\n🚀 Now you can run:")
    print("   python main_enhanced.py")
    print("   OR")
    print("   python main.py")

if __name__ == "__main__":
    create_all_filters()