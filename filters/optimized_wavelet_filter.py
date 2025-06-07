import cv2
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
