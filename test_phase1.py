#!/usr/bin/env python3
"""
Phase 1 Test Script

Simple test to verify the Phase 1 implementation of tensor metadata verification.
"""

import sys
import logging
import time
from deepstream_video_pipeline import DeepStreamVideoPipeline
from config import config

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('phase1_test.log')
    ]
)

def test_phase1():
    """Test Phase 1 tensor metadata verification"""
    print("🔍 Starting Phase 1 Test - Tensor Metadata Verification")
    
    # Use RTSP stream from config as originally intended
    rtsp_url = "rtsp://192.168.3.214:7447/jdr9oLlBkjyl3gDm?"
    
    try:
        # Create pipeline with Phase 2 simplified config
        pipeline = DeepStreamVideoPipeline(
            rtsp_url=rtsp_url,
            config=config,
            websocket_port=8765,
            preproc_config="config_preproc.txt"
        )
        
        print("✅ Pipeline created successfully")
        
        # Start pipeline
        if pipeline.start():
            print("✅ Pipeline started successfully")
            
            # Run for 30 seconds to capture metadata logs
            print("🔍 Running pipeline for 30 seconds to capture metadata logs...")
            start_time = time.time()
            
            while time.time() - start_time < 30:
                time.sleep(1)
                stats = pipeline.get_stats()
                if stats.get('frame_count', 0) > 0:
                    print(f"📊 Frame count: {stats.get('frame_count', 0)}")
                    
            print("✅ Test completed successfully")
            pipeline.stop()
            
        else:
            print("❌ Failed to start pipeline")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_phase1() 