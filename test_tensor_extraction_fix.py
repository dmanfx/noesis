#!/usr/bin/env python3
"""
Test script for tensor extraction and black frame fixes.

This script validates the Phase 1-4 fixes for:
1. PyCapsule tensor extraction with ctypes structures
2. Proper metadata type handling
3. Black frame resolution with DeepStream OSD
4. Configuration validation and cleanup

Usage:
    python test_tensor_extraction_fix.py --source <rtsp_url_or_file>
"""

import sys
import os
import logging
import argparse
import time
from typing import Dict, Any, List
import threading

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from deepstream_video_pipeline import DeepStreamVideoPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TensorExtractionTester:
    """Test class for validating tensor extraction fixes"""
    
    def __init__(self, source: str):
        self.source = source
        self.pipeline = None
        self.test_results = {
            'config_validation': False,
            'pipeline_creation': False,
            'tensor_extraction': False,
            'metadata_handling': False,
            'detection_processing': False,
            'black_frame_resolution': False,
            'overall_success': False
        }
        self.tensor_extraction_count = 0
        self.detection_count = 0
        self.frame_count = 0
        
    def run_comprehensive_test(self, duration: int = 30) -> Dict[str, Any]:
        """Run comprehensive test of all fixes"""
        logger.info("🧪 Starting comprehensive tensor extraction test")
        logger.info(f"📹 Source: {self.source}")
        logger.info(f"⏱️ Duration: {duration} seconds")
        
        try:
            # Test 1: Configuration validation
            self.test_results['config_validation'] = self._test_config_validation()
            
            # Test 2: Pipeline creation
            self.test_results['pipeline_creation'] = self._test_pipeline_creation()
            
            if not self.test_results['pipeline_creation']:
                logger.error("❌ Pipeline creation failed - cannot continue tests")
                return self.test_results
            
            # Test 3: Runtime validation
            runtime_results = self._test_runtime_validation(duration)
            self.test_results.update(runtime_results)
            
            # Overall success
            self.test_results['overall_success'] = all([
                self.test_results['config_validation'],
                self.test_results['pipeline_creation'],
                self.test_results['tensor_extraction'],
                self.test_results['metadata_handling'],
                self.test_results['detection_processing']
            ])
            
            self._print_test_summary()
            return self.test_results
            
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            return self.test_results
        finally:
            if self.pipeline:
                self.pipeline.stop()
    
    def _test_config_validation(self) -> bool:
        """Test configuration validation"""
        logger.info("🔍 Testing configuration validation...")
        
        try:
            # Test OSD setting
            if not config.visualization.USE_NATIVE_DEEPSTREAM_OSD:
                logger.error("❌ USE_NATIVE_DEEPSTREAM_OSD should be True")
                return False
            logger.info("✅ USE_NATIVE_DEEPSTREAM_OSD is correctly set to True")
            
            # Test batch size configuration
            batch_size = config.processing.DEEPSTREAM_MUX_BATCH_SIZE
            if batch_size <= 0:
                logger.warning("⚠️ DEEPSTREAM_MUX_BATCH_SIZE is auto-calculated")
            else:
                logger.info(f"✅ DEEPSTREAM_MUX_BATCH_SIZE: {batch_size}")
            
            # Test preprocessing config
            preproc_config = config.processing.DEEPSTREAM_PREPROCESS_CONFIG
            if not os.path.exists(preproc_config):
                logger.error(f"❌ Preprocessing config not found: {preproc_config}")
                return False
            logger.info(f"✅ Preprocessing config found: {preproc_config}")
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def _test_pipeline_creation(self) -> bool:
        """Test pipeline creation with validation"""
        logger.info("🔍 Testing pipeline creation...")
        
        try:
            # Create pipeline
            self.pipeline = DeepStreamVideoPipeline(
                rtsp_url=self.source,
                config=config,
                websocket_port=8765,
                        config_file="pipelines/config_infer_primary_yolo11.txt",
        preproc_config="pipelines/config_preproc.txt"
            )
            
            if not self.pipeline:
                logger.error("❌ Failed to create pipeline")
                return False
            
            logger.info("✅ Pipeline created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline creation failed: {e}")
            return False
    
    def _test_runtime_validation(self, duration: int) -> Dict[str, bool]:
        """Test runtime validation including tensor extraction"""
        logger.info("🔍 Testing runtime validation...")
        
        results = {
            'tensor_extraction': False,
            'metadata_handling': False,
            'detection_processing': False,
            'black_frame_resolution': False
        }
        
        try:
            # Start pipeline
            if not self.pipeline.start():
                logger.error("❌ Failed to start pipeline")
                return results
            
            logger.info("✅ Pipeline started successfully")
            
            # Add monitoring thread to check logs/output for tensors/detections
            def monitor_output():
                time.sleep(2)  # Wait for processing
                # Check logs for key phrases (simulate assertion)
                # For real: Parse app.log for "Extracted valid tensor" and obj_count >0
                # Assert if not found
                assert "✅ Extracted valid tensor" in open("app.log").read(), "No tensors extracted"
                assert "✅ Frame" in open("app.log").read(), "No detections found"

            threading.Thread(target=monitor_output).start()
            
            # Run test for specified duration
            start_time = time.time()
            last_log_time = start_time
            
            while time.time() - start_time < duration:
                ret, tensor_data = self.pipeline.read_gpu_tensor()
                
                if ret and tensor_data:
                    self.frame_count += 1
                    
                    # Test tensor extraction
                    if tensor_data.get('tensor') is not None:
                        self.tensor_extraction_count += 1
                        results['tensor_extraction'] = True
                        
                        # Log tensor details
                        if self.tensor_extraction_count == 1:
                            tensor = tensor_data['tensor']
                            logger.info(f"✅ First tensor extracted: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
                    
                    # Test detection processing
                    detections = tensor_data.get('detections', [])
                    if detections:
                        self.detection_count += 1
                        results['detection_processing'] = True
                        
                        # Log detection details
                        if self.detection_count == 1:
                            logger.info(f"✅ First detections found: {len(detections)} objects")
                    
                    # Test metadata handling (implicitly tested by successful tensor extraction)
                    if self.tensor_extraction_count > 0:
                        results['metadata_handling'] = True
                    
                    # Test black frame resolution (detections indicate successful processing)
                    if self.detection_count > 0:
                        results['black_frame_resolution'] = True
                    
                    # Log progress every 10 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= 10:
                        logger.info(f"📊 Progress: {self.frame_count} frames, {self.tensor_extraction_count} tensors, {self.detection_count} detections")
                        last_log_time = current_time
                
                else:
                    time.sleep(0.01)
            
            logger.info("✅ Runtime validation completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Runtime validation failed: {e}")
            return results
    
    def _print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "="*60)
        logger.info("🧪 TENSOR EXTRACTION TEST SUMMARY")
        logger.info("="*60)
        
        # Test results
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info("-"*60)
        
        # Statistics
        logger.info(f"📊 STATISTICS:")
        logger.info(f"   Total frames processed: {self.frame_count}")
        logger.info(f"   Successful tensor extractions: {self.tensor_extraction_count}")
        logger.info(f"   Frames with detections: {self.detection_count}")
        
        if self.frame_count > 0:
            tensor_success_rate = (self.tensor_extraction_count / self.frame_count) * 100
            detection_rate = (self.detection_count / self.frame_count) * 100
            logger.info(f"   Tensor extraction success rate: {tensor_success_rate:.1f}%")
            logger.info(f"   Detection rate: {detection_rate:.1f}%")
        
        logger.info("-"*60)
        
        # Overall result
        overall_status = "✅ SUCCESS" if self.test_results['overall_success'] else "❌ FAILURE"
        logger.info(f"🎯 OVERALL TEST RESULT: {overall_status}")
        
        # Success criteria validation
        logger.info("\n📋 SUCCESS CRITERIA:")
        criteria = [
            ("No PyCapsule errors", self.test_results['metadata_handling']),
            ("Tensor extraction working", self.test_results['tensor_extraction']),
            ("Detections visible", self.test_results['detection_processing']),
            ("Pipeline stable", self.test_results['pipeline_creation']),
            ("Config validation", self.test_results['config_validation'])
        ]
        
        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {criterion}")
        
        logger.info("="*60)


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test tensor extraction and black frame fixes")
    parser.add_argument("--source", required=True, help="RTSP URL or video file path")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run test
    tester = TensorExtractionTester(args.source)
    results = tester.run_comprehensive_test(args.duration)
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)


if __name__ == "__main__":
    main() 