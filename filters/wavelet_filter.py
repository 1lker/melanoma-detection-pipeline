import cv2
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
