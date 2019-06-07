import onnx
import numpy as np
import tvm
import tvm.relay as relay

onnx_model = onnx.load('resnet50sim.onnx')

num_class = 1000
data = np.random.uniform(-1, 1, size=(1,3,224,224)).astype("float32")

target = tvm.target.cuda()
ctx    = tvm.gpu()

shape_dict = {'0' : data.shape}
sym, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with relay.build_config(opt_level=1):
    intrp = relay.build_module.create_executor('graph',sym, ctx, target)

tvm_output = intrp.evaluate(sym)(tvm.nd.array(data.astype("float32")), **params).asnumpy()

print(tvm_output)
