import cv2
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
