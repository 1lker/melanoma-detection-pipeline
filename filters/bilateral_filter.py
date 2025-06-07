import cv2
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
