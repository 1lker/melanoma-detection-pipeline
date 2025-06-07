import cv2
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
