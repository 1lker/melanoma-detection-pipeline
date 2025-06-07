import cv2
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
