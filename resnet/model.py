import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, normalize=True):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, normalize=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()

        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion) if normalize else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, normalize=True):
        super().__init__()

        self.in_channels = 64
        self.normalize = normalize

        ## stem, dims 3x32x32 -> 64x32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        ## residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x, feat_vec=False):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        if feat_vec:
            return out
        return self.fc(out)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample_layers = [
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            ]
            if self.normalize:
                downsample_layers.append(nn.BatchNorm2d(out_channels * block.expansion))
            downsample = nn.Sequential(*downsample_layers)

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                normalize=self.normalize,
            )
        )
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    downsample=None,
                    normalize=self.normalize,
                )
            )
        return nn.Sequential(*layers)


def resnet18(num_classes=10, normalize=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, normalize=normalize)


def resnet34(num_classes=10, normalize=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, normalize=normalize)


def resnet50(num_classes=10, normalize=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, normalize=normalize)
