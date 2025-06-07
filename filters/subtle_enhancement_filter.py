import cv2
import numpy as np
from base_filter import BaseFilter

class SubtleEnhancementFilter(BaseFilter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.validate_image(image):
            return image
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            l_final = cv2.addWeighted(l, 0.8, l_enhanced, 0.2, 0)
            lab_enhanced = cv2.merge([l_final, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
            unsharp = cv2.addWeighted(enhanced, 1.05, gaussian, -0.05, 0)
            result = cv2.addWeighted(image, 0.7, unsharp, 0.3, 0)
            self.logger.debug('Applied subtle enhancement filter')
            return result
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            return image
    def get_name(self): return 'subtle_enhancement'
