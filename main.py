import os
import time
import logging
from typing import Dict, Any
from config import ConfigManager, PipelineConfig
from image_processor import ImageProcessor
from performance_monitor import PerformanceMonitor
from logger import PipelineLogger
from filter_factory import FilterFactory
from hair_removal import HairRemovalProcessor

# main.py - Fixed Main Application Entry Point
import os
import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all components
from config import ConfigManager, PipelineConfig
from logger import PipelineLogger
from image_processor import ImageProcessor
from performance_monitor import PerformanceMonitor

# Requirements content
REQUIREMENTS = """opencv-python>=4.8.0
numpy>=1.21.0
PyWavelets>=1.4.0
PyYAML>=6.0
psutil>=5.9.0
pathlib>=1.0.1
"""

# Default config content
CONFIG_YAML = """# Melanoma Detection Pipeline Configuration

processing:
  input_path: "./melanoma_cancer_dataset"
  output_path: "./processed_output"
  target_size: [224, 224]
  hair_kernel_size: [17, 17]
  hair_threshold: 10
  inpaint_radius: 1
  batch_size: 32
  num_workers: 4
  enable_gpu: false

filters:
  kernel_size: 5
  gaussian_sigma: 0.0
  bilateral_d: 9
  bilateral_sigma_color: 75.0
  bilateral_sigma_space: 75.0
  wavelet_type: 'haar'
  wavelet_levels: 1
  fourier_mask_size: 30

logging_level: 'INFO'
"""

def setup_files():
    """Setup configuration and requirements files if they don't exist"""
    
    # Create config file if it doesn't exist
    if not os.path.exists("config.yaml"):
        with open("config.yaml", "w") as f:
            f.write(CONFIG_YAML)
        print("✅ Created config.yaml file")
    
    # Create requirements file if it doesn't exist  
    if not os.path.exists("requirements.txt"):
        with open("requirements.txt", "w") as f:
            f.write(REQUIREMENTS)
        print("✅ Created requirements.txt file")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['cv2', 'numpy', 'pywt', 'yaml', 'psutil']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'pywt':
                import pywt
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def main():
    """Main application entry point"""
    print("🔬 Melanoma Detection Pipeline Starting...")
    print("=" * 50)
    
    # Setup files
    setup_files()
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 Install missing dependencies with:")
        print("   pip install -r requirements.txt")
        return
    
    try:
        # Load configuration
        config = ConfigManager.load_config()
        
        # Setup logger
        logger = PipelineLogger.setup_logger("Main", config.logging_level)
        
        # Initialize performance monitor
        monitor = PerformanceMonitor(logger)
        
        # Initialize processor
        processor = ImageProcessor(config)
        
        # Log startup information
        logger.info("🚀 Starting melanoma image processing pipeline")
        logger.info(f"📁 Input path: {config.processing.input_path}")
        logger.info(f"📁 Output path: {config.processing.output_path}")
        logger.info(f"🖼️  Target size: {config.processing.target_size}")
        logger.info(f"📦 Batch size: {config.processing.batch_size}")
        logger.info(f"👥 Workers: {config.processing.num_workers}")
        logger.info(f"🔧 GPU enabled: {config.processing.enable_gpu}")
        
        # Check if input directory exists
        if not os.path.exists(config.processing.input_path):
            logger.error(f"❌ Input directory does not exist: {config.processing.input_path}")
            print(f"\n❌ Input directory not found: {config.processing.input_path}")
            print("Please check your config.yaml and ensure the input_path is correct.")
            return
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Process dataset with all available filters
        print("\n🔄 Processing images...")
        results = processor.process_dataset()
        
        # Stop monitoring
        stats = monitor.stop_monitoring()
        
        # Log results
        print("\n📊 Processing Results:")
        print("=" * 30)
        total_processed = 0
        for filter_name, count in results.items():
            print(f"  {filter_name:15s}: {count:4d} images")
            logger.info(f"✅ {filter_name}: {count} images processed")
            total_processed += count
        
        print(f"\n🎯 Total images processed: {total_processed}")
        
        print("\n⚡ Performance Statistics:")
        print("=" * 30)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.2f}")
            else:
                print(f"  {key:20s}: {value}")
            logger.info(f"📈 {key}: {value}")
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Check results in: {config.processing.output_path}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
        logger.info("Pipeline interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()