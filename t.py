from dnnv.nn import Path, parse
op_graph = parse(Path("/home/c/thesis/benchmark/mk/cifar/onnx/cifar_net.onnx")).simplify()
model = op_graph.as_onnx()

# Get the serialized model as a byte string
model_bytes = model.SerializeToString()

# Write the model bytes to a file
with open("model.onnx", "wb") as f:
    f.write(model_bytes)
