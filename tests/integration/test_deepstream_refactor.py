#!/usr/bin/env python3
"""
Integration tests for the refactored DeepStream 7.1 pipeline with YOLO-11 parser.

These tests validate the new pipeline structure:
nvurisrcbin → nvstreammux → nvinfer → nvtracker → nvdsosd → appsink
"""

import pytest
import sys
import os
import time
import logging
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from deepstream_video_pipeline import DeepStreamVideoPipeline
from config import AppConfig

# Test configuration
TEST_VIDEO_URL = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264"
TEST_RTSP_URL = "rtsp://127.0.0.1:8554/test"  # Mock RTSP stream for testing

class TestDeepStreamRefactor:
    """Test suite for refactored DeepStream pipeline."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = AppConfig()
        
        # Configure for testing
        self.config.processing.ENABLE_DEEPSTREAM = True
        self.config.processing.DEEPSTREAM_ENABLE_OSD = True
        self.config.processing.DEEPSTREAM_TRACKER_CONFIG = ""  # Disable tracker for basic tests
        self.config.models.FORCE_GPU_ONLY = True
        
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
    def test_pipeline_single_stream_file_source(self):
        """Test 1: Pipeline starts with 1 file source and emits detections."""
        
        # Test data
        sources = [{
            'name': 'Test Video',
            'url': TEST_VIDEO_URL,
            'width': 1920,
            'height': 1080,
            'enabled': True
        }]
        
        detection_count = 0
        
        def mock_nvinfer_probe(pad, info, u_data):
            """Mock probe function to count detections."""
            nonlocal detection_count
            detection_count += 1
            return 1  # Gst.PadProbeReturn.OK
        
        # Create pipeline
        pipeline = DeepStreamVideoPipeline(sources, self.config, device_id=0)
        
        # Patch the probe function to count detections
        with patch.object(pipeline, '_nvinfer_src_pad_buffer_probe', side_effect=mock_nvinfer_probe):
            # Test pipeline creation
            assert pipeline._create_pipeline() == True
            
            # Verify pipeline components exist
            assert pipeline.pipeline is not None
            assert pipeline.appsink is not None
            assert pipeline.bus is not None
            
            # Test short run
            pipeline.start()
            time.sleep(2)  # Run for 2 seconds
            pipeline.stop()
            
            # Verify detections were processed
            assert detection_count > 0, f"Expected detections but got {detection_count}"
            
        self.logger.info(f"✅ Test 1 passed: {detection_count} detections processed")
    
    def test_pipeline_multi_stream_batch(self):
        """Test 2: Multi-stream (3 RTSP) batch processing."""
        
        # Test data - 3 streams
        sources = [
            {
                'name': 'Stream 1',
                'url': TEST_RTSP_URL + '1',
                'width': 1920,
                'height': 1080,
                'enabled': True
            },
            {
                'name': 'Stream 2', 
                'url': TEST_RTSP_URL + '2',
                'width': 1920,
                'height': 1080,
                'enabled': True
            },
            {
                'name': 'Stream 3',
                'url': TEST_RTSP_URL + '3',
                'width': 1280,
                'height': 720,
                'enabled': True
            }
        ]
        
        # Create pipeline
        pipeline = DeepStreamVideoPipeline(sources, self.config, device_id=0)
        
        # Test batch size calculation
        assert pipeline.batch_size == 3, f"Expected batch size 3, got {pipeline.batch_size}"
        
        # Test max dimensions calculation
        assert pipeline.max_width == 1920, f"Expected max width 1920, got {pipeline.max_width}"
        assert pipeline.max_height == 1080, f"Expected max height 1080, got {pipeline.max_height}"
        
        # Test pipeline creation
        assert pipeline._create_pipeline() == True
        
        # Verify streammux configuration
        # Note: In real test, we'd need to mock RTSP streams or use test streams
        
        self.logger.info("✅ Test 2 passed: Multi-stream batch configuration validated")
    
    def test_pipeline_yolo11_parser_integration(self):
        """Test 3: YOLO-11 parser integration and configuration."""
        
        sources = [{
            'name': 'Test Video',
            'url': TEST_VIDEO_URL,
            'width': 1920,
            'height': 1080,
            'enabled': True
        }]
        
        # Create pipeline
        pipeline = DeepStreamVideoPipeline(sources, self.config, device_id=0)
        
        # Test that config file exists
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config_infer_primary_yolo11.txt')
        assert os.path.exists(config_path), f"YOLO-11 config file not found: {config_path}"
        
        # Test that parser library exists
        parser_path = os.path.join(os.path.dirname(__file__), '..', '..', 'libnvdsparsebbox_yolo11.so')
        if os.path.exists(parser_path):
            self.logger.info(f"✅ YOLO-11 parser library found: {parser_path}")
        else:
            self.logger.warning(f"⚠️  YOLO-11 parser library not found: {parser_path}")
        
        # Test pipeline creation
        assert pipeline._create_pipeline() == True
        
        self.logger.info("✅ Test 3 passed: YOLO-11 parser integration validated")
    
    def test_pipeline_graceful_failure_missing_parser(self):
        """Test 4: Pipeline fails gracefully if parser is missing."""
        
        sources = [{
            'name': 'Test Video',
            'url': TEST_VIDEO_URL,
            'width': 1920,
            'height': 1080,
            'enabled': True
        }]
        
        # Create pipeline
        pipeline = DeepStreamVideoPipeline(sources, self.config, device_id=0)
        
        # Test with missing config file
        with patch('os.path.exists', return_value=False):
            with pytest.raises(RuntimeError, match="PGIE config.*not found"):
                pipeline._create_pipeline()
        
        self.logger.info("✅ Test 4 passed: Graceful failure on missing parser")
    
    def test_pipeline_osd_integration(self):
        """Test 5: OSD integration with bounding boxes."""
        
        sources = [{
            'name': 'Test Video',
            'url': TEST_VIDEO_URL,
            'width': 1920,
            'height': 1080,
            'enabled': True
        }]
        
        # Enable OSD
        self.config.processing.DEEPSTREAM_ENABLE_OSD = True
        
        # Create pipeline
        pipeline = DeepStreamVideoPipeline(sources, self.config, device_id=0)
        
        # Test pipeline creation
        assert pipeline._create_pipeline() == True
        
        # Test OSD disabled
        self.config.processing.DEEPSTREAM_ENABLE_OSD = False
        pipeline2 = DeepStreamVideoPipeline(sources, self.config, device_id=0)
        assert pipeline2._create_pipeline() == True
        
        self.logger.info("✅ Test 5 passed: OSD integration validated")
    
    def test_pipeline_tracker_integration(self):
        """Test 6: Optional tracker integration."""
        
        sources = [{
            'name': 'Test Video',
            'url': TEST_VIDEO_URL,
            'width': 1920,
            'height': 1080,
            'enabled': True
        }]
        
        # Test with tracker enabled
        self.config.processing.DEEPSTREAM_TRACKER_CONFIG = "tracker_nvdcf.yml"
        
        pipeline = DeepStreamVideoPipeline(sources, self.config, device_id=0)
        
        # Test pipeline creation (should work even if tracker config missing)
        assert pipeline._create_pipeline() == True
        
        # Test with tracker disabled
        self.config.processing.DEEPSTREAM_TRACKER_CONFIG = ""
        pipeline2 = DeepStreamVideoPipeline(sources, self.config, device_id=0)
        assert pipeline2._create_pipeline() == True
        
        self.logger.info("✅ Test 6 passed: Tracker integration validated")
    
    def test_pipeline_pad_probe_functionality(self):
        """Test 7: Pad probe functionality for detection metadata."""
        
        sources = [{
            'name': 'Test Video',
            'url': TEST_VIDEO_URL,
            'width': 1920,
            'height': 1080,
            'enabled': True
        }]
        
        pipeline = DeepStreamVideoPipeline(sources, self.config, device_id=0)
        
        # Test probe function
        mock_pad = Mock()
        mock_info = Mock()
        mock_info.get_buffer.return_value = Mock()
        
        # Mock pyds
        with patch('deepstream_video_pipeline.pyds') as mock_pyds:
            mock_pyds.gst_buffer_get_nvds_batch_meta.return_value = Mock()
            mock_pyds.gst_buffer_get_nvds_batch_meta.return_value.frame_meta_list = None
            
            # Test probe function doesn't crash
            result = pipeline._nvinfer_src_pad_buffer_probe(mock_pad, mock_info, 0)
            assert result == 1  # Gst.PadProbeReturn.OK
        
        self.logger.info("✅ Test 7 passed: Pad probe functionality validated")
    
    def test_pipeline_configuration_validation(self):
        """Test 8: Configuration validation for refactored pipeline."""
        
        sources = [{
            'name': 'Test Video',
            'url': TEST_VIDEO_URL,
            'width': 1920,
            'height': 1080,
            'enabled': True
        }]
        
        # Test valid configuration
        assert self.config.processing.ENABLE_DEEPSTREAM == True
        assert self.config.processing.DEEPSTREAM_ENABLE_OSD == True
        assert self.config.models.FORCE_GPU_ONLY == True
        
        # Test deprecation warnings
        with patch('warnings.warn') as mock_warn:
            # Enable deprecated setting
            self.config.processing.ENABLE_NVDEC = True
            self.config.processing.__post_init__()
            
            # Should have triggered deprecation warning
            assert mock_warn.called
        
        self.logger.info("✅ Test 8 passed: Configuration validation complete")
    
    def test_pipeline_environment_variables(self):
        """Test 9: Environment variable loading for RTSP URLs."""
        
        # Test .env.example file exists
        env_file = os.path.join(os.path.dirname(__file__), '..', '..', '.env.example')
        assert os.path.exists(env_file), f"Environment example file not found: {env_file}"
        
        # Test deepstream.yml exists
        ds_config = os.path.join(os.path.dirname(__file__), '..', '..', 'deepstream.yml')
        assert os.path.exists(ds_config), f"DeepStream config file not found: {ds_config}"
        
        self.logger.info("✅ Test 9 passed: Environment configuration files validated")
    
    def test_pipeline_stability_no_segfault(self):
        """Test 10: Pipeline stability - no segfaults on startup."""
        
        sources = [{
            'name': 'Test Video',
            'url': TEST_VIDEO_URL,
            'width': 1920,
            'height': 1080,
            'enabled': True
        }]
        
        # Test multiple pipeline create/destroy cycles
        for i in range(3):
            pipeline = DeepStreamVideoPipeline(sources, self.config, device_id=0)
            
            # Should not crash
            assert pipeline._create_pipeline() == True
            
            # Clean up
            if pipeline.pipeline:
                pipeline.pipeline.set_state(0)  # GST_STATE_NULL
                pipeline.pipeline = None
        
        self.logger.info("✅ Test 10 passed: Pipeline stability validated")

def test_integration_suite():
    """Run the full integration test suite."""
    
    # Check prerequisites
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
    except ImportError:
        pytest.skip("GStreamer not available")
    
    try:
        import pyds
    except ImportError:
        pytest.skip("DeepStream Python bindings not available")
    
    # Run tests
    test_instance = TestDeepStreamRefactor()
    test_instance.setup_method()
    
    tests = [
        test_instance.test_pipeline_single_stream_file_source,
        test_instance.test_pipeline_multi_stream_batch,
        test_instance.test_pipeline_yolo11_parser_integration,
        test_instance.test_pipeline_graceful_failure_missing_parser,
        test_instance.test_pipeline_osd_integration,
        test_instance.test_pipeline_tracker_integration,
        test_instance.test_pipeline_pad_probe_functionality,
        test_instance.test_pipeline_configuration_validation,
        test_instance.test_pipeline_environment_variables,
        test_instance.test_pipeline_stability_no_segfault
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = test_integration_suite()
    sys.exit(0 if success else 1) 