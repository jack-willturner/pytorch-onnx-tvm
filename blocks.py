import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class StandardBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class GConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, group_split, stride=1, kernel_size=3, padding=1, bias=False):
        super(GConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=bottleneck//group_split)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out

class AConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, groups, stride=1, kernel_size=3, padding=1, bias=False):
        super(AConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=groups)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out
