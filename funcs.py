import torch
import torch.nn.functional as F
from models import *
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator

## or, as an iterable:
def get_block_dict():
    blocks   = {'S'         : Conv,
                'S-2x2'     : Conv2x2,
                'G(2)'      : DConvA2,
                'G(4)'      : DConvA4,
                'G(8)'      : DConvA8,
                'G(16)'     : DConvA16,
                'G(N/16)'   : DConvG16,
                'G(N/8)'    : DConvG8,
                'G(N/4)'    : DConvG4,
                'G(N/2)'    : DConvG2,
                'G(N)'      : DConv,
                'B(2)'      : ConvB2,
                'B(4)'      : ConvB4,
                'BG(2,2)'   : A2B2,
                'BG(2,4)'   : A4B2,
                'BG(2,8)'   : A8B2,
                'BG(2,16)'  : A16B2,
                'BG(2,M/16)': G16B2,
                'BG(2,M/8)' : G8B2,
                'BG(2,M/4)' : G4B2,
                'BG(2,M/2)' : G2B2,
                'BG(2,M)'   : DConvB2,
                'BG(4,M)'   : DConvB4}
    return blocks
