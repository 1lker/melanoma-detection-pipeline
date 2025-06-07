import cv2
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
