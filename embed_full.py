import onnx
from pathlib import Path

def embed_external_data(model_path, embedded_path):
    model = onnx.load(model_path, load_external_data=True)
    # Embed initializers
    for init in model.graph.initializer:
        if len(init.external_data) > 0:
            data_file = Path(model_path).parent / init.external_data[0].key
            with open(data_file, 'rb') as f:
                init.raw_data = f.read()
            init.external_data[:] = []  # Clear repeated field
    # Embed constant node attributes
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                t = attr.t
                if len(t.external_data) > 0:
                    data_file = Path(model_path).parent / t.external_data[0].key
                    with open(data_file, 'rb') as f:
                        t.raw_data = f.read()
                    t.external_data[:] = []  # Clear repeated field
    onnx.save(model, embedded_path)
    print('Full embedded ONNX saved to', embedded_path)

if __name__ == '__main__':
    embed_external_data('ma_onnx_out/model.onnx', 'ma_onnx_out/model_full_embedded.onnx')
