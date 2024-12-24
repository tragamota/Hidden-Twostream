import torch
from torch import nn

from MotionNet.MotionNet import MotionNet
from Resnet import ResNet, BasicBlock


class TemporalNet(nn.Module):

    def __init__(self, motionnet_weights=None, block_type=BasicBlock, layers=[2, 2, 2, 2], num_classes=101):
        super(TemporalNet, self).__init__()

        self.motionNet = MotionNet()
        self.motion_upscale = nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1)
        self.temporal = ResNet(block_type, input_channels=20, layers=layers, num_classes=num_classes, use_classifier=True)

        if motionnet_weights is not None:
            self.motionNet.load_state_dict(torch.load(motionnet_weights, weights_only=True))


    def forward(self, x):
        flow = self.motionNet(x)[0]
        flow = self.motion_upscale(flow)
        # flow = torch.clip(flow, -20/255, 20/255)

        return self.temporal(flow)