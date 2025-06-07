import cv2
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
