# libcustom2d_preprocess.so Source Code Documentation

## Overview
`libcustom2d_preprocess.so` is a DeepStream custom preprocessing library that handles tensor preparation for 2D image processing. It's used by the `nvdspreprocess` GStreamer plugin to prepare input tensors for neural network inference.

## Source Code Location
The source code is located in the DeepStream SDK at:
```
opt/nvidia/deepstream/deepstream-7.1/sources/gst-plugins/gst-nvdspreprocess/nvdspreprocess_lib/
```

## Key Files

### 1. Main Library File
**File**: `nvdspreprocess_lib.cpp` (223 lines)
**Purpose**: Main library implementation with exported functions

### 2. Header File
**File**: `nvdspreprocess_lib.h` (83 lines)
**Purpose**: Function declarations and interface definitions

### 3. Implementation File
**File**: `nvdspreprocess_impl.cpp` (578 lines)
**Purpose**: Core tensor preparation and CUDA operations

### 4. Implementation Header
**File**: `nvdspreprocess_impl.h` (225 lines)
**Purpose**: Class definitions and implementation interfaces

## Exported Functions

### 1. `CustomTensorPreparation`
```cpp
NvDsPreProcessStatus CustomTensorPreparation(
    CustomCtx *ctx, 
    NvDsPreProcessBatch *batch, 
    NvDsPreProcessCustomBuf *&buf,
    CustomTensorParams &tensorParam, 
    NvDsPreProcessAcquirer *acquirer
);
```
**Purpose**: Main tensor preparation function called by the plugin
**Functionality**:
- Acquires buffer from tensor pool
- Calls tensor implementation to prepare tensor
- Synchronizes CUDA stream
- Updates batch size in tensor parameters

### 2. `CustomTransformation`
```cpp
NvDsPreProcessStatus CustomTransformation(
    NvBufSurface *in_surf, 
    NvBufSurface *out_surf, 
    CustomTransformParams &params
);
```
**Purpose**: Synchronous surface transformation
**Functionality**:
- Sets session parameters for NvBufSurfTransform
- Performs batched transformation
- Handles error conditions

### 3. `CustomAsyncTransformation`
```cpp
NvDsPreProcessStatus CustomAsyncTransformation(
    NvBufSurface *in_surf, 
    NvBufSurface *out_surf, 
    CustomTransformParams &params
);
```
**Purpose**: Asynchronous surface transformation
**Functionality**:
- Sets session parameters for NvBufSurfTransform
- Performs async batched transformation
- Uses sync objects for completion tracking

### 4. `initLib`
```cpp
CustomCtx *initLib(CustomInitParams initparams);
```
**Purpose**: Library initialization
**Functionality**:
- Parses configuration parameters
- Sets up normalization and mean subtraction
- Initializes tensor implementation
- Returns custom context

### 5. `deInitLib`
```cpp
void deInitLib(CustomCtx *ctx);
```
**Purpose**: Library cleanup
**Functionality**:
- Deallocates custom context
- Cleans up resources

## Key Classes

### 1. `CustomCtx`
```cpp
struct CustomCtx {
    CustomInitParams initParams;
    CustomMeanSubandNormParams custom_mean_norm_params;
    std::unique_ptr<NvDsPreProcessTensorImpl> tensor_impl;
};
```
**Purpose**: Main context structure holding library state

### 2. `NvDsPreProcessTensorImpl`
**Purpose**: Core tensor preparation implementation
**Key Methods**:
- `prepare_tensor()`: Main tensor preparation
- `setScaleOffsets()`: Set normalization parameters
- `setMeanFile()`: Set mean image file
- `allocateResource()`: Allocate CUDA resources
- `syncStream()`: Synchronize CUDA stream

### 3. `CudaStream`
**Purpose**: CUDA stream management
**Functionality**:
- Stream creation with priority
- Automatic cleanup on destruction

### 4. `CudaDeviceBuffer`
**Purpose**: CUDA device memory management
**Functionality**:
- Device memory allocation
- Automatic cleanup on destruction

## Configuration Parameters

### From config_preproc.txt:
- `pixel-normalization-factor`: Normalization factor (default: 0.003921568)
- `offsets`: Channel offsets for mean subtraction (e.g., "0;0;0")
- `mean-file`: Path to mean image file (PPM format)

### Example Configuration:
```ini
[user-configs]
pixel-normalization-factor=0.003921568
offsets=0;0;0
```

## Tensor Preparation Process

1. **Buffer Acquisition**: Get buffer from tensor pool
2. **Surface Transformation**: Scale and convert input surfaces
3. **Normalization**: Apply pixel normalization factor
4. **Mean Subtraction**: Subtract channel means (if specified)
5. **Format Conversion**: Convert to required tensor format (NCHW/NHWC)
6. **Memory Copy**: Copy to output tensor buffer
7. **Synchronization**: Wait for CUDA operations to complete

## CUDA Operations

### Memory Management
- Device memory allocation for mean data
- CUDA stream management for async operations
- Automatic cleanup of CUDA resources

### Kernels Used
- Surface transformation via NvBufSurfTransform
- Memory copy operations
- Format conversion operations

## Error Handling

### Status Codes
- `NVDSPREPROCESS_SUCCESS`: Operation successful
- `NVDSPREPROCESS_TENSOR_NOT_READY`: Tensor not ready
- `NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED`: Transformation failed
- `NVDSPREPROCESS_CUDA_ERROR`: CUDA operation failed
- `NVDSPREPROCESS_CONFIG_FAILED`: Configuration error

### Error Recovery
- Buffer release on failure
- CUDA error checking and reporting
- Graceful degradation

## Usage in Your Pipeline

### Current Configuration
Your `config_preproc.txt` uses:
```ini
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so
custom-tensor-preparation-function=CustomTensorPreparation
```

### Parameters Applied
- **Normalization**: `0.003921568` (1/255 for uint8 to float conversion)
- **Offsets**: `0;0;0` (no mean subtraction)
- **Input Shape**: `1;3;640;640` (batch;channels;height;width)
- **Format**: NCHW (network-input-order=0)

## Building the Library

### Prerequisites
- CUDA toolkit
- DeepStream SDK
- GStreamer development headers

### Build Process
```bash
cd opt/nvidia/deepstream/deepstream-7.1/sources/gst-plugins/gst-nvdspreprocess/nvdspreprocess_lib/
make
```

### Output
- `libcustom2d_preprocess.so`: Shared library
- Installed to `/opt/nvidia/deepstream/deepstream/lib/gst-plugins/`

## Customization Options

### 1. Modify Normalization
Change `pixel-normalization-factor` in config:
```ini
pixel-normalization-factor=0.017507  # For different normalization
```

### 2. Add Mean Subtraction
Add offsets in config:
```ini
offsets=123.675;116.280;103.53  # ImageNet means
```

### 3. Use Mean File
Specify mean image file:
```ini
mean-file=/path/to/mean.ppm
```

### 4. Custom Implementation
Create your own library with same interface:
- Implement required functions
- Follow same function signatures
- Build as shared library
- Update config to point to your library

## Performance Considerations

### Optimization Features
- CUDA stream-based async operations
- Memory pooling for tensor buffers
- Batch processing support
- GPU-accelerated transformations

### Memory Management
- Automatic CUDA resource cleanup
- Buffer pooling to reduce allocations
- Efficient memory layout for tensor operations

## Troubleshooting

### Common Issues
1. **Library not found**: Check path in config file
2. **CUDA errors**: Verify GPU memory availability
3. **Format mismatches**: Check network input order and shape
4. **Mean file errors**: Verify PPM format and resolution

### Debug Options
- Enable debug logging in GStreamer
- Check CUDA error codes
- Verify tensor dimensions and format
- Monitor GPU memory usage

## License
NVIDIA Proprietary - See SPDX headers in source files 