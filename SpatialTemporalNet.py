from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=stride,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(output_channels)
        )

        self.downsample = downsample

    def forward(self, x):
        residual = x

        x = self.block(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, input_channels, output_channels, stride, downsample=None):
        super(BottleneckBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=stride,bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels * self.expansion, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(output_channels)
        )

        self.downsample = downsample

    def forward(self, x):
        residual = x

        x = self.block(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, input_channels, layers, num_classes, use_classifier=False):
        super(ResNet, self).__init__()

        self.use_classifier = use_classifier

        self.head = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = nn.Sequential(
            *self._make_layer(block, 64, layers[0]),
            *self._make_layer(block, 128, layers[1], stride=2),
            *self._make_layer(block, 256, layers[2], stride=2),
            *self._make_layer(block, 512, layers[3], stride=2)
        )

        if use_classifier:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")

    def _make_layer(self, block_type, channels, block_count, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != channels * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block_type.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels * block_type.expansion)
            )

        layers.append(block_type(self.base_layers, channels, stride, downsample=downsample))

        for _ in range(1, block_count):
            layers.append(block_type(self.base_layers, channels))

        self.base_layers = channels * block_type.expansion

        return layers

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)

        if self.use_classifier:
            x = self.avgpool(x)
            x = self.flatten()
            x = self.fc(x)

        return x
