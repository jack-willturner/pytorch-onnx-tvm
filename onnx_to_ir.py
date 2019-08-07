import onnx
import numpy as np
import tvm
import tvm.relay as relay
import torch
import torch.nn as nn

x = torch.randn(1,32,224,224).numpy()

s_conv  = nn.Conv2d(32,64,3)
g_conv  = nn.Conv2d(32,64,3,groups=4)
b_conv  = nn.Conv2d(32,32,3)
bg_conv = nn.Conv2d(32,32,3,groups=4)

convs = [(s_conv,'standard'), (g_conv,'group'), (b_conv,'bottle'), (bg_conv,'bottlegroup')]

for conv, name in convs:
    onnx_model = onnx.load('blocks/'+name+'.t7')
    target = 'llvm'

    input_name = '1'
    shape_dict = {input_name: x.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    with relay.build_config(opt_level=1):
        intrp = relay.build_module.create_executor('graph', mod, tvm.cpu(0), target)

    dtype = 'float32'
    tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
