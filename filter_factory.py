import logging
from typing import Dict, Type, List
from base_filter import BaseFilter
from config import FilterConfig

from filters.average_blur_filter import AverageBlurFilter
from filters.gaussian_filter import GaussianFilter
from filters.fourier_filter import FourierFilter
from filters.wavelet_filter import WaveletFilter
from filters.bilateral_filter import BilateralFilter

try:
    from filters.improved_bilateral_filter import ImprovedBilateralFilter
    from filters.subtle_enhancement_filter import SubtleEnhancementFilter
except ImportError:
    ImprovedBilateralFilter = None
    SubtleEnhancementFilter = None

class FilterFactory:
    _filters = {
        'average_blur': AverageBlurFilter,
        'gaussian': GaussianFilter,
        'fourier': FourierFilter,
        'wavelet': WaveletFilter,
        'bilateral': BilateralFilter,
    }
    
    if ImprovedBilateralFilter:
        _filters['improved_bilateral'] = ImprovedBilateralFilter
    if SubtleEnhancementFilter:
        _filters['subtle_enhancement'] = SubtleEnhancementFilter
    
    @classmethod
    def create_filter(cls, filter_name, config, logger):
        if filter_name not in cls._filters:
            raise ValueError(f'Unknown filter: {filter_name}')
        return cls._filters[filter_name](config, logger)
    
    @classmethod
    def get_available_filters(cls):
        return list(cls._filters.keys())
    
    @classmethod
    def get_recommended_filters(cls):
        recommended = ['average_blur', 'gaussian', 'bilateral']
        if 'improved_bilateral' in cls._filters:
            recommended.append('improved_bilateral')
        if 'subtle_enhancement' in cls._filters:
            recommended.append('subtle_enhancement')
        return recommended
