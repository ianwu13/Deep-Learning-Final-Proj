import torch
import torch.nn as nn

# Adapted from torchvision implementation of resnet and FPN


class resblock(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, downsample=None, groups = 1, base_width = 64, dilation = 1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class resnet_fpn(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        # each element in the tuple indicates if we should replace
        # the 2x2 stride with a dilated convolution instead
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2, dilate=False)
        self.layer3 = self._make_layer(256, 6, stride=2, dilate=False)
        self.layer4 = self._make_layer(512, 3, stride=2, dilate=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.upsample = nn.Upsample(scale_factor=2)
        self.decoder1 = nn.Conv2d(512, 256, 3, 1, 1)
        self.decoder2 = nn.Conv2d(512, 128, 3, 1, 1)
        self.decoder3 = nn.Conv2d(256, 64, 3, 1, 1)
        self.decoder4 = nn.Conv2d(128, 64, 3, 1, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def _make_layer(self, channels, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(channels),
            )

        layers = []
        layers.append(
            resblock(
                self.inplanes, channels, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = channels
        for _ in range(1, blocks):
            layers.append(
                resblock(
                    self.inplanes,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x0 = self.layer1(x)
        x1 = self.layer2(x0)
        x2 = self.layer3(x1)
        x = self.layer4(x2)

        x = self.upsample(self.decoder1(x))
        x = torch.cat((x, x2), dim=1)
        x = self.upsample(self.decoder2(x))
        x = torch.cat((x, x1), dim=1)
        x = self.upsample(self.decoder3(x))
        x = torch.cat((x, x0), dim=1)
        x = self.decoder4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
