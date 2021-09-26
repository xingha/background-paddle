"""
@Filename : model.py
@Time : 2021/09/09 15:49:37
@Author : zhoubishu
@Email : zhoubs11@chinaunicom.cn
@Descript : 
"""

from paddle import nn
# from paddle.fluid import layers
from paddle.vision.models.resnet import ResNet, BottleneckBlock


class ResNetEncoder(ResNet):
    layers = {
        "resnet50": 50,
        "resnet101": 101,
    }

    def __init__(self, in_channels, norm_layer=None, variant='resnet50'):
        super().__init__(block=BottleneckBlock,
                         depth=self.layers[variant],
                         )
        if in_channels != 3:
            self.conv1 = nn.Conv2D(in_channels, 64, 7, 2, 3, bias_attr=False)

        del self.avgpool
        del self.fc

    def forward(self, x):
        x0 = x  # 1/1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x  # 1/4
        x = self.layer2(x)
        x3 = x  # 1/8
        x = self.layer3(x)
        x = self.layer4(x)
        x4 = x  # 1/16
        return x4, x3, x2, x1, x0
