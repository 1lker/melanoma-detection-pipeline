import cv2
import numpy as np
from base_filter import BaseFilter

class AverageBlurFilter(BaseFilter):
    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.validate_image(image):
            return image
        try:
            kernel_size = self.config.kernel_size
            blurred = cv2.blur(image, (kernel_size, kernel_size))
            self.logger.debug(f'Applied average blur with kernel size {kernel_size}')
            return blurred
        except Exception as e:
            self.logger.error(f'Error: {str(e)}')
            return image
    def get_name(self): return 'average_blur'
