import cv2
import numpy as np
from base_filter import BaseFilter

class ImprovedBilateralFilter(BaseFilter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.validate_image(image):
            return image
        try:
            filtered = cv2.bilateralFilter(image, 9, 40, 40)
            result = cv2.addWeighted(image, 0.4, filtered, 0.6, 0)
            self.logger.debug('Applied improved bilateral filter')
            return result
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            return image
    def get_name(self): return 'improved_bilateral'
