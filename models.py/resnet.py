"""yolo backend
using resnet101
"""

import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU(inplace=True)

        self.short_cut = nn.Sequential()
        if stride!=1 or input_channels != BasicBlock.expansion * output_channels:
            self.short_cut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )


    def forward(self, x):
        residual = x
        residual = self.short_cut(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        output = self.relu(x + residual)
        return output

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()

        #zero mapping
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 
                        stride=stride, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3,
                        padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels * BottleNeck.expansion, 1, bias=False),
            nn.BatchNorm2d(BottleNeck.expansion * output_channels)
        )

        self.relu = nn.ReLU(inplace=True)

        #identity mapping
        self.short_cut = nn.Sequential()

        if stride != 1 or input_channels != BottleNeck.expansion * output_channels:
            self.short_cut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels * BottleNeck.expansion, stride, bias=False),
                nn.BatchNorm2d(output_channels * BottleNeck.expansion)
            )


    def forward(self, x):
        residual = self.short_cut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        print(x.shape)
        print(residual.shape)
        output = self.relu(residual + x)
        return output

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=20):
        super().__init__()

        self.input_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.input_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_channels, self.input_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_channels, self.input_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.conv2 = self._make_layers(block, 64, num_block[0], 2)
        self.conv3 = self._make_layers(block, 128, num_block[1], 2)
        self.conv4 = self._make_layers(block, 256, num_block[2], 2)
        self.conv5 = self._make_layers(block, 512, num_block[3], 2)

        self.feature = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )


    def forward(self, x):
        x = self.feature(x)
        return x

    
    def _make_layers(self, block, output_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for s in strides:
            layers.append(block(self.input_channels, output_channels, s))
            self.input_channels = output_channels * block.expansion
        
        return nn.Sequential(*layers)




        





from torch.autograd import Variable
import torch
bb = BasicBlock(3, 30)
bb(Variable(torch.Tensor(1, 3, 32, 32)))

bn = BottleNeck(3, 30)
print(bn(Variable(torch.Tensor(1, 3, 32, 32))))

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

resnet = resnet101()
print(resnet(Variable(torch.Tensor(1, 3, 100, 100))))
