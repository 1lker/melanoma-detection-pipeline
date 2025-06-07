# Fixed main_enhanced.py
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from logger import PipelineLogger
from performance_monitor import PerformanceMonitor
from filter_factory import FilterFactory

# Try to import enhanced processor, fallback to basic
try:
    from image_processor_enhanced import StructurePreservingImageProcessor as ImageProcessor
    print("✅ Using enhanced structure-preserving processor")
except ImportError:
    try:
        from image_processor import ImageProcessor
        print("⚠️  Using basic processor (no structure preservation)")
    except ImportError:
        print("❌ No image processor found!")
        sys.exit(1)

def main():
    """Enhanced main with structure preservation and medical filters"""
    print("🏥 Enhanced Melanoma Detection Pipeline")
    print("=" * 60)
    
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
        logger.info("🚀 Starting enhanced melanoma processing pipeline")
        logger.info(f"📁 Input path: {config.processing.input_path}")
        logger.info(f"📁 Output path: {config.processing.output_path}")
        
        # Show available filters
        try:
            all_filters = FilterFactory.get_available_filters()
            medical_filters = FilterFactory.get_medical_filters()
            logger.info(f"🎯 Available filters: {all_filters}")
            logger.info(f"🏥 Medical filters: {medical_filters}")
        except Exception:
            logger.warning("Could not get filter information")
        
        # Check input directory
        if not os.path.exists(config.processing.input_path):
            logger.error(f"❌ Input directory not found: {config.processing.input_path}")
            print(f"\n❌ Input directory not found: {config.processing.input_path}")
            print("Please check your config.yaml file")
            return
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Choose filters to run
        try:
            medical_filters = FilterFactory.get_medical_filters()
            if medical_filters:
                print(f"\n🔄 Processing with enhanced medical filters: {medical_filters}")
                filter_list = medical_filters
            else:
                print(f"\n🔄 Processing with basic filters")
                filter_list = ['average_blur', 'gaussian', 'bilateral']
        except Exception:
            print(f"\n🔄 Processing with default filters")
            filter_list = ['average_blur', 'gaussian']
        
        # Process dataset
        if hasattr(processor, 'process_dataset_structured'):
            print("✅ Using structure-preserving processing")
            results = processor.process_dataset_structured(filter_list)
        else:
            print("⚠️  Using basic processing")
            results = processor.process_dataset(filter_list)
        
        # Stop monitoring
        stats = monitor.stop_monitoring()
        
        # Display results
        print("\n📊 Processing Results:")
        print("=" * 40)
        
        if isinstance(results, dict) and results:
            # Check if we have structured results
            first_result = next(iter(results.values()))
            if isinstance(first_result, dict):
                # Structure-preserving results
                for filter_name, filter_results in results.items():
                    print(f"\n🎯 {filter_name.upper()}:")
                    total_for_filter = 0
                    for category, count in filter_results.items():
                        print(f"   {category:20s}: {count:4d} images")
                        total_for_filter += count
                    print(f"   {'TOTAL':20s}: {total_for_filter:4d} images")
            else:
                # Basic results
                for filter_name, count in results.items():
                    print(f"  {filter_name:20s}: {count:4d} images")
        else:
            print("No results to display")
        
        # Performance stats
        if stats:
            print(f"\n⚡ Performance Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:.2f}")
                else:
                    print(f"  {key:20s}: {value}")
        
        print(f"\n✅ Enhanced pipeline completed!")
        print(f"📁 Results in: {config.processing.output_path}")
        
        # Show expected structure
        print(f"\n📁 Expected Output Structure:")
        print(f"   processed_output/")
        for filter_name in filter_list:
            print(f"   ├── {filter_name}/")
            print(f"   │   ├── train/benign/")
            print(f"   │   ├── train/malignant/")
            print(f"   │   ├── test/benign/")
            print(f"   │   └── test/malignant/")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()