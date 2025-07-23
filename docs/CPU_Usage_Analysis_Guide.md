# CPU Usage Analysis Guide

## Overview

This guide provides comprehensive instructions for analyzing and optimizing the CPU usage in your GPU-only video processing application. The system currently shows 35-40% CPU utilization despite being designed for GPU-only processing.

## Current Architecture Analysis

### Key Components Contributing to CPU Usage

1. **Multi-threading/Multi-processing Architecture**
   - ApplicationManager coordinates multiple components
   - FrameProcessor/StreamingPipeline for video processing
   - Separate threads for frame reading, processing, and streaming
   - Analysis process for decoupled pipeline processing

2. **Data Movement and Serialization**
   - Inter-process communication via multiprocessing.Queue
   - WebSocket broadcasting (JSON serialization every 1 second)
   - Frame data copying between processes and threads

3. **Processing Components**
   - DetectionManager (YOLO + pose estimation + feature extraction)
   - TrackingSystem (ByteTrack with IoU calculations)
   - VisualizationManager (frame annotation and rendering)

4. **Communication Overhead**
   - WebSocket server broadcasting stats and frames
   - Queue management and monitoring
   - Shared data synchronization

## CPU Profiling System

Your application already includes a comprehensive CPU profiling system:

### Existing Profiling Components

- **ThreadCPUMonitor**: Monitors individual thread CPU usage
- **SystemResourceMonitor**: System-wide resource monitoring
- **FunctionProfiler**: Function-level timing and CPU usage
- **ComprehensiveCPUProfiler**: Main coordinator class

### Profiling Integration

The profiling system is integrated into:
- Main application (`main.py`)
- Processing pipelines (`pipeline.py`)
- All major processing functions via `@profile_function` decorators

## How to Run CPU Analysis

### Step 1: Enable Profiling

Profiling is now enabled in `config.py`:
```python
ENABLE_PROFILING: bool = True  # Enable profiling to identify CPU usage sources
```

### Step 2: Start Your Application

```bash
python main.py
```

### Step 3: Run CPU Analysis (in another terminal)

```bash
# Using the helper script
python run_cpu_analysis.py

# Or directly
python cpu_analysis.py --duration 120 --interval 1.0
```

### Step 4: Review Results

Analysis results will be saved to `cpu_analysis_results/`:

- **`summary_stats.json`**: Overall CPU usage statistics
- **`process_analysis.json`**: Per-process CPU breakdown
- **`thread_analysis.json`**: Per-thread CPU breakdown  
- **`optimization_recommendations.md`**: Specific optimization suggestions

## Expected CPU Usage Sources

Based on the architecture analysis, the 35-40% CPU usage likely comes from:

### 1. Threading Overhead (Estimated: 8-12%)
- Multiple concurrent threads per camera
- Thread context switching
- Synchronization overhead

### 2. Data Serialization/Movement (Estimated: 10-15%)
- JSON serialization for WebSocket broadcasting
- Frame data copying between processes
- Queue operations and memory management

### 3. Tracking Algorithms (Estimated: 5-8%)
- ByteTrack IoU calculations
- Track management and history maintenance
- Feature matching and comparison

### 4. WebSocket Broadcasting (Estimated: 3-5%)
- Real-time stats broadcasting (every 1 second)
- Frame streaming and compression
- Client connection management

### 5. Visualization Processing (Estimated: 4-7%)
- Frame annotation and rendering
- Keypoint drawing and trace visualization
- Mask overlay processing

### 6. Queue Management (Estimated: 3-5%)
- Inter-process queue operations
- Memory allocation and deallocation
- Queue size monitoring

## Optimization Strategies

### Immediate Actions (Low Risk)

1. **Reduce WebSocket Broadcast Frequency**
   ```python
   # In websocket_server.py, change from 1.0 to 2.0 seconds
   await self._periodic_stats_broadcast(interval_seconds=2.0)
   ```

2. **Increase Frame Skipping**
   ```python
   # In config.py
   FRAME_SKIP: int = 5  # Process every 6th frame instead of every 4th
   ```

3. **Reduce Queue Sizes**
   ```python
   # In config.py
   MAX_QUEUE_SIZE: int = 20  # Reduce from 30
   ```

### Medium-term Optimizations (Medium Risk)

1. **Optimize Thread Pool Usage**
   - Replace individual threads with ThreadPoolExecutor
   - Reduce number of concurrent threads

2. **Implement Binary WebSocket Protocol**
   - Replace JSON with binary serialization
   - Use MessagePack or Protocol Buffers

3. **Optimize Tracking Algorithm**
   - Reduce IoU calculation frequency
   - Implement early termination for tracking

### Advanced Optimizations (Higher Risk)

1. **Implement Zero-Copy Frame Sharing**
   - Use shared memory for frame data
   - Reduce memory copying overhead

2. **Asynchronous Processing Pipeline**
   - Replace multiprocessing with asyncio where possible
   - Reduce process creation overhead

3. **GPU-Accelerated Visualization**
   - Move frame annotation to GPU
   - Use CUDA kernels for drawing operations

## Monitoring and Validation

### Key Metrics to Track

1. **System CPU Usage**: Target < 20%
2. **Per-Process CPU**: Identify highest consumers
3. **Thread CPU Distribution**: Balance thread workload
4. **Memory Usage**: Monitor for memory leaks
5. **Queue Sizes**: Prevent bottlenecks

### Validation Process

1. Run baseline analysis (current state)
2. Apply one optimization at a time
3. Re-run analysis to measure impact
4. Document changes and rollback if needed

## Real-time Monitoring

The application provides real-time CPU profiling data via WebSocket:

```javascript
// WebSocket client can access CPU profiling data
{
  "type": "stats",
  "payload": {
    "cpu_profiling": {
      "threads": {...},
      "system": {...},
      "functions": {...}
    }
  }
}
```

## Troubleshooting

### High CPU Usage Patterns

1. **Sustained 30-40% CPU**: Normal for current architecture
2. **Spikes > 60%**: Check for memory pressure or I/O bottlenecks
3. **Gradual increase over time**: Potential memory leak

### Common Issues

1. **Analysis script fails**: Ensure `psutil` is installed
2. **No data collected**: Verify application is running during analysis
3. **Permission errors**: Run with appropriate user permissions

## Next Steps

1. **Run initial analysis** to establish baseline
2. **Identify top 3 CPU consumers** from results
3. **Apply targeted optimizations** based on findings
4. **Measure improvement** with follow-up analysis
5. **Document successful optimizations** for future reference

## Files Created

- `cpu_analysis.py`: Comprehensive CPU analysis tool
- `run_cpu_analysis.py`: Helper script for running analysis
- `config.py`: Updated to enable profiling
- `docs/CPU_Usage_Analysis_Guide.md`: This documentation

The comprehensive CPU profiling system is now ready to help you identify and optimize the sources of CPU usage in your video processing application. 