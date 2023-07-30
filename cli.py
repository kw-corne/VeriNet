import ast
import shlex
import signal
import subprocess
import sys

import onnxruntime

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
    dnnv_simplify = sys.argv[8] == "True"
    transpose_matmul_weights = sys.argv[9] == "True"

    config = ast.literal_eval(config)

    for k, v in config.items():
        setattr(CONFIG, k, v)

    if max_procs == -1:
        max_procs = None

    onnx_parser = ONNXParser(
        network,
        transpose_fc_weights=transpose_matmul_weights,
        use_64bit=False,
        dnnv_simplify=dnnv_simplify,
    )

    new_input_shape = onnx_parser.get_simplified_input_shape()

    if new_input_shape:
        input_shape = new_input_shape[0]

    model = onnx_parser.to_pytorch()
    model.eval()
    vnnlib_parser = VNNLIBParser(property)
    objectives = vnnlib_parser.get_objectives_from_vnnlib(model, input_shape)

    solver = VeriNet(use_gpu=use_gpu, max_procs=max_procs)

    # HACK: `solver.cleanup()` can hang forever. Dont have time to find the
    # root cause right now, so solving it in a hacky way.
    def _cleanup():
        cmd = "pkill -f multiprocessing.spawn"
        subprocess.run(shlex.split(cmd))
        cmd = "pkill -f multiprocessing.forkserver"
        subprocess.run(shlex.split(cmd))
        solver.cleanup()

    def _handler(*_):
        raise Exception("Cleanup timed out")

    signal.signal(signal.SIGALRM, _handler)

    try:
        # TODO: Loop em
        status = solver.verify(objective=objectives[0], timeout=timeout)
        print("STATUS: ", status)
    finally:
        signal.alarm(1)
        try:
            _cleanup()
        except Exception as e:
            print(e)
