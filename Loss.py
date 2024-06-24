import torch
from torch import nn


class CharbonnierPenalty(nn.Module):

    def __init__(self):
        super(CharbonnierPenalty, self).__init__()

    def forward(self, x, epsilon=1e-3, alpha=0.4):
        return (x ** 2 + epsilon ** 2) ** alpha


class PixelwiseReconstructionLoss(nn.Module):

    def __init__(self):
        super(PixelwiseReconstructionLoss, self)

    def forward(self):
        return 0


class SmoothnessLoss(nn.Module):
    charbonnier_penalty = CharbonnierPenalty()

    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, flow_x, flow_y):
        delta_Vx_x = torch.gradient(flow_x, axis=1)
        delta_Vx_y = torch.gradient(flow_x, axis=2)
        delta_Vy_x = torch.gradient(flow_y, axis=1)
        delta_Vy_y = torch.gradient(flow_y, axis=2)

        return (self.charbonnier_penalty(delta_Vx_x) + self.charbonnier_penalty(delta_Vx_y) +
                self.charbonnier_penalty(delta_Vy_x) + self.charbonnier_penalty(delta_Vy_y))


class SSIMLoss(nn.Module):

    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self):
        return 0


class MotionNetLoss(nn.Module):
    SSIMLoss = SSIMLoss()

    def __init__(self, loss_weights=[]):
        super(MotionNetLoss, self).__init__()

    def forward(self):

        return 0
