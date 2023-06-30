import ast
import sys

from verinet.parsers.onnx_parser import ONNXParser
from verinet.parsers.vnnlib_parser import VNNLIBParser
from verinet.util.config import CONFIG
from verinet.verification.verinet import VeriNet

if __name__ == "__main__":
    network = sys.argv[1]
    property = sys.argv[2]
    timeout = int(sys.argv[3])
    config = sys.argv[4]
    input_shape = ast.literal_eval(sys.argv[5])
    max_procs = int(sys.argv[6])
    use_gpu = sys.argv[7] == "True"

    config = ast.literal_eval(config)

    for k, v in config.items():
        setattr(CONFIG, k, v)

    if max_procs == -1:
        max_procs = None

    onnx_parser = ONNXParser(
        network,
        transpose_fc_weights=False,
        use_64bit=False,
    )
    model = onnx_parser.to_pytorch()
    model.eval()

    vnnlib_parser = VNNLIBParser(property)
    objectives = vnnlib_parser.get_objectives_from_vnnlib(model, input_shape)

    solver = VeriNet(use_gpu=use_gpu, max_procs=max_procs)

    try:
        status = solver.verify(objective=objectives[0], timeout=timeout)
        print("STATUS: ", status)
    finally:
        solver.cleanup()
