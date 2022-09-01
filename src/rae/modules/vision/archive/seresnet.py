import logging
from typing import Dict, List, Sequence, Type

import torch
from torch import Tensor, nn
from torchvision.models.resnet import conv1x1, conv3x3

from rae.data.vision.datamodule import MetaData
from rae.modules.enumerations import Output

pylogger = logging.getLogger(__name__)


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False), nn.ReLU(inplace=True), nn.Linear(c // r, c, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, r=16):
        super(SEBasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.se = SE_Block(planes, r)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # add SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        input_size: int,
        input_channels: int,
        n_classes: int,
        layers: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.inplanes = 64

        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(SEBasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(SEBasicBlock, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(SEBasicBlock, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(SEBasicBlock, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * SEBasicBlock.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[SEBasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        latents = x = torch.flatten(x, 1)
        x = self.fc(x)

        return {
            Output.LOGITS: x,
            Output.DEFAULT_LATENT: latents,
            Output.BATCH_LATENT: latents,
        }

    def set_finetune_mode(self) -> None:
        pass
