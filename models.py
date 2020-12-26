import torch.hub
import torch.nn as nn

from torchvision.models.video.resnet import BasicBlock, R2Plus1dStem, Conv2Plus1D

model_location = {
    "r2plus1d_34_8_ig65m": "/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth",
    "r2plus1d_34_32_ig65m": "r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth",
    "r2plus1d_34_8_kinetics": "/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth",
    "r2plus1d_34_32_kinetics": "r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth",
}

def r2plus1d_34_8_ig65m(num_classes, pretrained=False):
    """R(2+1)D 34-layer IG65M model for clips of length 8 frames.

    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads weights pretrained on 65 million Instagram videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 487, "pretrained on 487 classes"
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_8_ig65m",
                       pretrained=pretrained)


def r2plus1d_34_32_ig65m(num_classes, pretrained=False):
    """R(2+1)D 34-layer IG65M model for clips of length 32 frames.

    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads weights pretrained on 65 million Instagram videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 359, "pretrained on 359 classes"
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_32_ig65m",
                       pretrained=pretrained)


def r2plus1d_34_8_kinetics(num_classes, pretrained=False):
    """R(2+1)D 34-layer IG65M-Kinetics model for clips of length 8 frames.

    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads IG65M weights fine-tuned on Kinetics videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 400, "pretrained on 400 classes"
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_8_kinetics",
                       pretrained=pretrained)


def r2plus1d_34_32_kinetics(num_classes, pretrained=False):
    """R(2+1)D 34-layer IG65M-Kinetics model for clips of length 32 frames.

    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads IG65M weights fine-tuned on Kinetics videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 400, "pretrained on 400 classes"
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_32_kinetics",
                       pretrained=pretrained)


def r2plus1d_34(num_classes, pretrained=False, arch=None):
    model = VideoResNet(block=BasicBlock,
                        conv_makers=[Conv2Plus1D] * 4,
                        layers=[3, 4, 6, 3],
                        stem=R2Plus1dStem)

    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)

    # Fix difference in PyTorch vs Caffe2 architecture
    # https://github.com/facebookresearch/VMZ/issues/89
    # https://github.com/pytorch/vision/issues/1265
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.load(model_location[arch])
        model.load_state_dict(state_dict)

    return model
class VideoResNet(nn.Module):
    #改版，最后是一个sigmoid输出
    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return torch.sigmoid(x)

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
