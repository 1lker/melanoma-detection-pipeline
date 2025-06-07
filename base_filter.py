from abc import ABC, abstractmethod
import logging
from config import FilterConfig
import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

class BaseFilter(ABC):
    """Abstract base class for all image filters"""
    
    def __init__(self, config: FilterConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the filter to an image"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the filter name for logging and file naming"""
        pass
    
    def validate_image(self, image: np.ndarray) -> bool:
        """Validate input image"""
        if image is None:
            self.logger.error("Input image is None")
            return False
        
        if len(image.shape) not in [2, 3]:
            self.logger.error(f"Invalid image shape: {image.shape}")
            return False
            
        return True
