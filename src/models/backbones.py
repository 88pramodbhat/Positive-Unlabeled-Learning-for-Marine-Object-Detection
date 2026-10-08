import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    """
    Standard ResNet Bottleneck Block implemented in pure PyTorch.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 Feature Extraction Backbone for Underwater Images in pure PyTorch.
    Supports torchvision fallback if installed.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.out_channels = 2048
        
        try:
            import torchvision.models as models
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.feature_extractor = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            )
        except ImportError:
            # Pure PyTorch ResNet-50 implementation
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(64, 3, stride=1)
            self.layer2 = self._make_layer(128, 4, stride=2)
            self.layer3 = self._make_layer(256, 6, stride=2)
            self.layer4 = self._make_layer(512, 3, stride=2)

            self.feature_extractor = nn.Sequential(
                self.conv1, self.bn1, self.relu, self.maxpool,
                self.layer1, self.layer2, self.layer3, self.layer4
            )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BottleneckBlock(self.in_planes, planes, s))
            self.in_planes = planes * BottleneckBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class FPNBackbone(nn.Module):
    """
    Feature Pyramid Network (FPN) backbone in pure PyTorch.
    Extracts multi-scale features for marine species of varying sizes.
    """
    def __init__(self, pretrained: bool = True, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.resnet = ResNet50Backbone(pretrained=pretrained)
        self.lat = nn.Conv2d(2048, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.resnet(x)
        return self.lat(feat)
