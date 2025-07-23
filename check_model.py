#!/usr/bin/env python3
import onnx
from collections import Counter

model = onnx.load('models/yolo11m.onnx')

print('Input names:')
for input in model.graph.input:
    print(f'  {input.name}')

print('\nOutput names:')
for output in model.graph.output:
    print(f'  {output.name}')

print(f'\nModel IR version: {model.ir_version}')
print(f'Opset version: {model.opset_import[0].version}')
print(f'Producer: {model.producer_name}')
print(f'Total nodes: {len(model.graph.node)}')
print(f'Total initializers: {len(model.graph.initializer)}')

# Check for any unsupported ops
print('\nFirst10 node types:')
for i, node in enumerate(model.graph.node[:10]):
    print(f'  {i}: {node.op_type}')

# Count all node types
node_types = [node.op_type for node in model.graph.node]
type_counts = Counter(node_types)
print('\nAll node types:')
for op_type, count in type_counts.items():
    print(f'  {op_type}: {count}')

# Check TensorRT version
try:
    import tensorrt as trt
    print(f'\nTensorRT version: {trt.__version__}')
    
    # Test TensorRT ONNX parsing
    print('\nTesting TensorRT ONNX parsing...')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open('models/yolo11m.onnx', 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('❌ TensorRT failed to parse ONNX model')
            for error in range(parser.num_errors):
                print(f'  Error {error}: {parser.get_error(error)}')
        else:
            print('✅ TensorRT parsed ONNX model successfully')
            print(f'  Network layers: {network.num_layers}')
            
            # Check input dimensions
            print('\nInput dimensions:')
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                print(f'  {input_tensor.name}: {input_tensor.shape}')
                
                # Check for dynamic dimensions
                for j, dim in enumerate(input_tensor.shape):
                    if dim < 0:
                        print(f'    ⚠️  Dynamic dimension at index {j}: {dim}')
                    elif dim == 0:
                        print(f'    ⚠️  Zero dimension at index {j}: {dim}')
                        
except ImportError:
    print('\nTensorRT not available in Python')

# Detailed input shape analysis
print('\nDetailed input shape analysis:')
for input in model.graph.input:
    print(f'\nInput: {input.name}')
    shape = input.type.tensor_type.shape
    print(f'  Dimensions: {shape.dim}')
    
    for i, dim in enumerate(shape.dim):
        if dim.dim_param:
            print(f'    Dim {i}: Dynamic ({dim.dim_param})')
        elif dim.dim_value < 0:
            print(f'    Dim {i}: Invalid negative value ({dim.dim_value})')
        elif dim.dim_value == 0:
            print(f'    Dim {i}: Zero value ({dim.dim_value})')
        else:
            print(f'    Dim {i}: Fixed value ({dim.dim_value})') 