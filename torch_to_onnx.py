import torch
import torch.onnx
import torch.nn as nn
from models import *
import pandas as pd
from count_ops import *

### first we export the whole model to onnx
resnet50 = resnet50()
dummy_input = torch.randn(1,3,224,224)
_ = resnet50(dummy_input)
torch.onnx.export(resnet50, dummy_input, 'resnet/resnet50.onnx')

# store a list of parameter counts and spatial dimensions for each layer
df = []

print(resnet50)

i = 0
### iterate over layers and export to onnx
for block in resnet50.children():

    if isinstance(block, nn.Conv2d):
        fname = 'resnet/resnet50_' + str(i) + '.onnx'
        ops, params = measure_layer(block, dummy_input)
        df.append([block.in_channels, block.out_channels, block.kernel_size,
                        block.stride, block.padding, block.bias, dummy_input.size()[2], dummy_input.size()[3], ops, params])
        torch.onnx.export(block, dummy_input, fname)
        i += 1

    elif isinstance(block, nn.Sequential):
        for layer in block:
            fname = 'resnet/resnet50_' + str(i) + '.onnx'
            dummy_input = torch.randn(1,layer.conv1.in_channels,layer.conv1_input_size[2],layer.conv1_input_size[3])
            ops, params = measure_layer(layer.conv1,dummy_input)
            df.append([fname,layer.conv1.in_channels, layer.conv1.out_channels, layer.conv1.kernel_size,
                            layer.conv1.stride, layer.conv1.padding, layer.conv1.bias, layer.conv1_input_size[2],
                            layer.conv1_input_size[3], ops, params])
            torch.onnx.export(layer.conv1, dummy_input,fname)
            i += 1

            fname = 'resnet/resnet50_' + str(i) + '.onnx'
            dummy_input = torch.randn(1,layer.conv2.in_channels,layer.conv2_input_size[2],layer.conv2_input_size[3])
            ops, params = measure_layer(layer.conv2, dummy_input)
            df.append([fname,layer.conv2.in_channels, layer.conv2.out_channels, layer.conv2.kernel_size,
                            layer.conv2.stride, layer.conv2.padding, layer.conv2.bias, layer.conv2_input_size[2],
                            layer.conv2_input_size[3], ops, params])
            torch.onnx.export(layer.conv2, dummy_input, fname)
            i += 1

            fname = 'resnet/resnet50_' + str(i) + '.onnx'
            dummy_input =  torch.randn(1,layer.conv3.in_channels,layer.conv3_input_size[2],layer.conv3_input_size[3])
            ops, params = measure_layer(layer.conv3, dummy_input)
            df.append([fname,layer.conv3.in_channels, layer.conv3.out_channels, layer.conv3.kernel_size,
                            layer.conv3.stride, layer.conv3.padding, layer.conv3.bias, layer.conv3_input_size[2],
                            layer.conv3_input_size[3], ops, params])
            torch.onnx.export(layer.conv3,dummy_input, fname)
            i += 1

    elif isinstance(block, nn.Linear):
        #df.append([block.in_features, block.out_features, None, None, None, block.bias, None, None])
        torch.onnx.export(block, torch.randn(1,block.in_features), ('resnet/resnet50_linear.onnx'))
        i += 1

df = pd.DataFrame(df, columns=['filename','in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias', 'input_spatial_x', 'input_spatial_y', 'ops', 'params'])
df.to_csv('resnet/layer_info.csv')
