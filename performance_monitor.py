import psutil
import time
from typing import Dict, Any
import logging

class PerformanceMonitor:
    """Monitor system performance during processing"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = None
        self.start_memory = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        
        self.logger.info(f"Performance monitoring started")
        self.logger.info(f"Initial memory usage: {self.start_memory / 1024**3:.2f} GB")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics"""
        if self.start_time is None:
            return {}
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        duration = end_time - self.start_time
        memory_diff = (end_memory - self.start_memory) / 1024**3
        cpu_percent = psutil.cpu_percent()
        
        stats = {
            'duration_seconds': duration,
            'memory_change_gb': memory_diff,
            'cpu_percent': cpu_percent,
            'peak_memory_gb': psutil.virtual_memory().used / 1024**3
        }
        
        self.logger.info(f"Processing completed in {duration:.2f}s")
        self.logger.info(f"Memory change: {memory_diff:+.2f} GB")
        self.logger.info(f"CPU usage: {cpu_percent:.1f}%")
        
        return stats