import logging
from typing import Tuple
from config import ProcessingConfig
import numpy as np
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass



class HairRemovalProcessor:
    """Hair detection and removal for dermoscopic images"""
    
    def __init__(self, config: ProcessingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def detect_and_remove_hair(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detect and remove hair from dermoscopic image
        
        Returns:
            Tuple of (processed_image, has_hair)
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Create morphological kernel
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, 
                self.config.hair_kernel_size
            )
            
            # Apply blackhat morphological operation
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            # Apply threshold
            _, threshold = cv2.threshold(
                blackhat, 
                self.config.hair_threshold, 
                255, 
                cv2.THRESH_BINARY
            )
            
            # Check if hair is present
            hair_pixels = cv2.countNonZero(threshold)
            has_hair = hair_pixels > 0
            
            if has_hair:
                # Apply inpainting to remove hair
                inpainted = cv2.inpaint(
                    image, 
                    threshold, 
                    self.config.inpaint_radius, 
                    cv2.INPAINT_TELEA
                )
                
                self.logger.debug(f"Hair detected and removed. Hair pixels: {hair_pixels}")
                return inpainted, True
            else:
                self.logger.debug("No hair detected")
                return image, False
                
        except Exception as e:
            self.logger.error(f"Error in hair removal: {str(e)}")
            return image, False
