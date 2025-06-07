import yaml
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from pathlib import Path

@dataclass
class FilterConfig:
    """Configuration for individual filters"""
    kernel_size: int = 5
    gaussian_sigma: float = 0.0
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    wavelet_type: str = 'haar'
    wavelet_levels: int = 1
    fourier_mask_size: int = 30

@dataclass
class ProcessingConfig:
    """Main processing configuration"""
    input_path: str
    output_path: str
    target_size: Tuple[int, int] = (224, 224)
    hair_kernel_size: Tuple[int, int] = (17, 17)
    hair_threshold: int = 10
    inpaint_radius: int = 1
    batch_size: int = 32
    num_workers: int = 4
    enable_gpu: bool = False
    
@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    processing: ProcessingConfig
    filters: FilterConfig
    logging_level: str = 'INFO'
    
class ConfigManager:
    """Manages configuration loading and validation"""
    
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> PipelineConfig:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return ConfigManager._dict_to_config(config_dict)
        else:
            return ConfigManager.get_default_config()
    
    @staticmethod
    def get_default_config() -> PipelineConfig:
        """Get default configuration"""
        return PipelineConfig(
            processing=ProcessingConfig(
                input_path="./melanoma_cancer_dataset",
                output_path="./processed_output"
            ),
            filters=FilterConfig()
        )
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> PipelineConfig:
        """Convert dictionary to configuration objects"""
        processing_dict = config_dict.get('processing', {})
        filters_dict = config_dict.get('filters', {})
        
        processing = ProcessingConfig(**processing_dict)
        filters = FilterConfig(**filters_dict)
        
        return PipelineConfig(
            processing=processing,
            filters=filters,
            logging_level=config_dict.get('logging_level', 'INFO')
        )