from base_filter import BaseFilter
import cv2
import numpy as np
from config import FilterConfig
import logging
from typing import Optional, Tuple
from abc import abstractmethod
from dataclasses import dataclass



class AverageBlurFilter(BaseFilter):
    """Average blur filter implementation"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply average blur filter"""
        if not self.validate_image(image):
            return image
            
        try:
            kernel_size = self.config.kernel_size
            blurred = cv2.blur(image, (kernel_size, kernel_size))
            
            self.logger.debug(f"Applied average blur with kernel size {kernel_size}")
            return blurred
            
        except Exception as e:
            self.logger.error(f"Error applying average blur: {str(e)}")
            return image
    
    def get_name(self) -> str:
        return "average_blur"