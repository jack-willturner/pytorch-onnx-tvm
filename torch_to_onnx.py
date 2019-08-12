import torch
import torch.onnx
import torch.nn as nn
from models import *
import pandas as pd
from count_ops import *
import argparse

parser = argparse.ArgumentParser(description='AutoTVM from ONNX checkpoints')
parser.add_argument('--model', default='resnet50', type=str)
args = parser.parse_args()

dummy_input = torch.randn(1,3,224,224)
if args.model == 'resnet50':
    net = resnet50()
    base_file = 'resnet/resnet50'
    save_folder = ['resnet']
elif args.model == 'vgg16':
    net = vgg16()
    base_file = 'vgg/vgg16'
    save_folder = 'vgg'

_ = net(dummy_input)
torch.onnx.export(net, dummy_input, str(base_file + '.onnx'))

# store a list of parameter counts and spatial dimensions for each layer
df = []

print(net)

i = 0
### iterate over layers and export to onnx
for block in net.children():
    if isinstance(block, nn.Conv2d):
        fname = base_file + '_' + str(i) + '.onnx'
        ops, params = measure_layer(block, dummy_input)
        df.append([fname,block.in_channels, block.out_channels, block.kernel_size,
                        block.stride, block.padding, block.bias is not None, dummy_input.size()[2], dummy_input.size()[3], ops, params])
        torch.onnx.export(block, dummy_input, fname)
        i += 1
        if args.model == 'vgg16':
            dummy_input = block(dummy_input)

    elif isinstance(block, nn.Sequential):
        for layer in block:
            fname = base_file + '_' + str(i) + '.onnx'
            dummy_input = torch.randn(1,layer.conv1.in_channels,layer.conv1_input_size[2],layer.conv1_input_size[3])
            ops, params = measure_layer(layer.conv1,dummy_input)
            df.append([fname,layer.conv1.in_channels, layer.conv1.out_channels, layer.conv1.kernel_size,
                            layer.conv1.stride, layer.conv1.padding, layer.conv1.bias, int(layer.conv1_input_size[2]),
                            int(layer.conv1_input_size[3]), ops, params])
            torch.onnx.export(layer.conv1, dummy_input,fname)
            i += 1

            fname = base_file + '_' + str(i) + '.onnx'
            dummy_input = torch.randn(1,layer.conv2.in_channels,layer.conv2_input_size[2],layer.conv2_input_size[3])
            ops, params = measure_layer(layer.conv2, dummy_input)
            df.append([fname,layer.conv2.in_channels, layer.conv2.out_channels, layer.conv2.kernel_size,
                            layer.conv2.stride, layer.conv2.padding, layer.conv2.bias, int(layer.conv2_input_size[2]),
                            int(layer.conv2_input_size[3]), ops, params])
            torch.onnx.export(layer.conv2, dummy_input, fname)
            i += 1

            fname = base_file + '_' + str(i) + '.onnx'
            dummy_input =  torch.randn(1,layer.conv3.in_channels,layer.conv3_input_size[2],layer.conv3_input_size[3])
            ops, params = measure_layer(layer.conv3, dummy_input)
            df.append([fname,layer.conv3.in_channels, layer.conv3.out_channels, layer.conv3.kernel_size,
                            layer.conv3.stride, layer.conv3.padding, layer.conv3.bias, int(layer.conv3_input_size[2]),
                            int(layer.conv3_input_size[3]), ops, params])
            torch.onnx.export(layer.conv3,dummy_input, fname)
            i += 1

    elif isinstance(block, nn.Linear):
        fname = base_file + '_l_' + str(i) + '.onnx'
        torch.onnx.export(block, torch.randn(1,block.in_features), (fname))
        i += 1

df = pd.DataFrame(df, columns=['filename','in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'bias', 'input_spatial_x', 'input_spatial_y', 'ops', 'params'])
df.to_csv(save_folder+'/layer_info.csv')
