import torch
import onnx
from funcs import *
from models import *

dummy_input = torch.randn(1,32,224,224)

s_conv  = nn.Conv2d(32,64,3)
g_conv  = nn.Conv2d(32,64,3,groups=4)
b_conv  = nn.Conv2d(32,32,3)
bg_conv = nn.Conv2d(32,32,3,groups=4)

convs = [(s_conv,'standard'), (g_conv,'group'), (b_conv,'bottle'), (bg_conv,'bottlegroup')]

for conv,name in convs:
    model = conv
    _ = model(dummy_input)
    torch.onnx.export(model, dummy_input, 'blocks/'+name+'.t7')
