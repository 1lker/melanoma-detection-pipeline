import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from logger import PipelineLogger
from image_processor import ImageProcessor
from performance_monitor import PerformanceMonitor
from filter_factory import FilterFactory

def main():
    print('🏥 Melanoma Detection Pipeline - Structure Preserving')
    print('=' * 60)
    
    try:
        config = ConfigManager.load_config()
        logger = PipelineLogger.setup_logger('Main', config.logging_level)
        monitor = PerformanceMonitor(logger)
        processor = ImageProcessor(config)
        
        logger.info('🚀 Starting structure-preserving pipeline')
        logger.info(f'📁 Input: {config.processing.input_path}')
        logger.info(f'📁 Output: {config.processing.output_path}')
        
        recommended_filters = FilterFactory.get_recommended_filters()
        logger.info(f'🌟 Recommended filters: {recommended_filters}')
        
        if not os.path.exists(config.processing.input_path):
            logger.error(f'❌ Input directory not found: {config.processing.input_path}')
            return
        
        monitor.start_monitoring()
        
        print(f'🔄 Processing with: {recommended_filters}')
        print('📁 Preserving structure: train/benign, train/malignant, test/benign, test/malignant')
        
        results = processor.process_dataset(recommended_filters)
        stats = monitor.stop_monitoring()
        
        print('📊 Results by Category:')
        print('=' * 60)
        
        if results:
            for filter_name, filter_results in results.items():
                print(f'🎯 {filter_name.upper()}:')
                total = 0
                for category, count in filter_results.items():
                    print(f'   {category:20s}: {count:4d} images')
                    total += count
                print(f'   {TOTAL:20s}: {total:4d} images')
        
        if stats:
            print(f'⚡ Performance:')
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f'  {key:20s}: {value:.2f}')
                else:
                    print(f'  {key:20s}: {value}')
        
        print(f'✅ Pipeline completed!')
        print(f'📁 Results: {config.processing.output_path}')
        print(f'📁 Structure Created:')
        print(f'   processed_output/')
        for f in recommended_filters:
            print(f'   ├── {f}/')
            print(f'   │   ├── train/benign/ & train/malignant/')
            print(f'   │   └── test/benign/ & test/malignant/')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        raise

if __name__ == '__main__':
    main()
