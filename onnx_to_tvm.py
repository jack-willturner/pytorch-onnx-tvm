import onnx
import numpy as np
import tvm
import tvm.relay as relay
import argparse

parser = argparse.ArgumentParser(description='AutoTVM from ONNX checkpoints')
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

def get_network(filename, batch_size, in_channels, in_x, in_y):
    onnx_model = onnx.load(filename)
    data = np.random.uniform(-1, 1, size=(batch_size,in_channels,in_x,in_y)).astype("float32")
    shape_dict = {'0' : data.shape}
    sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return sym, params, data

if args.cpu:
    target = 'llvm'
    ctx = tvm.cpu()
else:
    target = tvm.target.cuda()
    ctx    = tvm.gpu()


sym, params, data = get_network('resnet/resnet50_1.onnx', 1, 64, 56, 56)
with relay.build_config(opt_level=1):
    intrp = relay.build_module.create_executor('graph',sym, ctx, target)

tvm_output = intrp.evaluate(sym)(tvm.nd.array(data.astype("float32")), **params).asnumpy()

print(tvm_output)
