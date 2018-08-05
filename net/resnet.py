"""yolo backend
resnet 

Author: baiyu
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
                nn.Conv2d(input_channels, output_channels * BottleNeck.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(output_channels * BottleNeck.expansion)
            )


    def forward(self, x):
        residual = self.short_cut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        output = self.relu(residual + x)
        return output

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=20):
        super().__init__()
        self.num_classes = num_classes

        self.input_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.input_channels, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_channels, self.input_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_channels, self.input_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = self._make_layers(block, 64, num_block[0], 2)
        self.conv3 = self._make_layers(block, 128, num_block[1], 2)
        self.conv4 = self._make_layers(block, 256, num_block[2], 2)
        self.conv5 = self._make_layers(block, 512, num_block[3], 2)

        self.avgpool = nn.AvgPool2d(2, stride=2)

        #I reduced the parameters to a quarter 
        #otherwise It would be over 400 billion parameters
        #now we have over 100 billion parameters
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, int(self.input_channels / 4), 1, bias=False),
            nn.BatchNorm2d(int(self.input_channels / 4)),
            nn.ReLU(inplace=True)
        )

    #2 fc layers:
    #"""Our detection network has 24 convolutional layers followed by 
    #2 fully connected layers."""
    def _fc_layer(self, input_channels):
        fc = nn.Sequential(
            nn.Linear(input_channels, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, (10 + self.num_classes) * 7 * 7)
        )

        return fc

    def forward(self, x):
        #feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.reduce_conv(x)
        x = self.avgpool(x)

        flat_length = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1, flat_length)
        x = self._fc_layer(flat_length)(x)
        x = x.view(-1, 30, 7, 7)

        return x

    def _make_layers(self, block, output_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for s in strides:
            layers.append(block(self.input_channels, output_channels, s))
            self.input_channels = output_channels * block.expansion
        
        return nn.Sequential(*layers)








def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


