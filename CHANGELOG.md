# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **CRITICAL**: Eliminated dual-CUDA-context bug causing "invalid resource handle" and "cuTensor permute execute failed" errors
  - Modified `CUDAContextManager` to use `cuda.Context.attach()` instead of creating new context
  - Context now shares PyTorch's primary CUDA context, preventing resource conflicts
  - `push_context()` and `pop_context()` methods converted to lightweight no-ops for backward compatibility
  - Removed explicit context cleanup operations that could interfere with PyTorch's context management

### Migration Note
**Dual-CUDA-Context Bug Fix**

If you were experiencing "invalid resource handle" or "cuTensor permute execute failed" errors, this release eliminates those issues by implementing a shared-context design:

- **What Changed**: `CUDAContextManager` now attaches to PyTorch's primary CUDA context instead of creating its own
- **Impact**: All CUDA operations now share the same context, preventing resource handle conflicts
- **Backward Compatibility**: Existing code using `push_context()` and `pop_context()` will continue to work unchanged
- **Testing**: Run your single-camera pipeline for >1 minute to verify the fix

No code changes are required in your application - the fix is internal to the context manager.

## [Unreleased] - PyNvVideoCodec Ingest Path Polish

### Fixed
- **Corrected colour conversion API**: Updated `nvdec_rtsp_gpu_reader.py` to use the proper PyNvVideoCodec `PySurfaceConverter` API instead of the deprecated `ColorspaceConversionContext`
- **Eliminated redundant GPU operations**: Removed unnecessary `.cuda()` calls and device transfers since `SurfaceTensor` already returns GPU tensors
- **Zero-copy DLPack integration**: Implemented DLPack tensor format for zero-copy hand-off between PyNvVideoCodec reader and DALI pipeline
- **DALI version detection**: Added automatic detection of DALI >= 1.30.0 for native DLPack support with CuPy fallback for older versions

### Added
- **Configuration tuning knobs**: Added `GPU_READER_QUEUE_SIZE` and `GPU_READER_MAX_CONSECUTIVE_FAILURES` settings to `ProcessingSettings` for runtime tuning
- **Enhanced error handling**: Improved error messages for missing dependencies and version compatibility issues
- **DLPack test coverage**: Added comprehensive unit tests for DLPack conversion paths and configuration integration

### Changed
- **Surface converter initialization**: Now uses proper width/height from decoder and explicit pixel format enums
- **Failure threshold**: Made consecutive failure limit configurable instead of hardcoded
- **DALI external source**: Optimized for zero-copy operation with automatic fallback for older DALI versions

### Technical Details
- Updated `_initialize_decoder()` to use `PySurfaceConverter(w, h, NV12, RGB, gpu_id)` 
- Implemented DLPack tensor queueing with `dlpack.to_dlpack(tensor)` in decode loop
- Added DALI version detection: `DALI_VER >= (1, 30, 0)` for DLPack support
- Wired configuration values through factory function and pipeline creation
- Enhanced unit tests with mocked DLPack operations and config integration

### Performance Impact
- Eliminated extra GPU memory copies in colorspace conversion
- Reduced CPU overhead through zero-copy DLPack tensor transfers
- Improved error recovery with configurable failure thresholds
- Optimized DALI pipeline integration for modern DALI versions 