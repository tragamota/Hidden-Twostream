import torch
from torch import nn
import torch.nn.functional as F

from WarpTransform import WarpTransformation


class CharbonnierPenalty(nn.Module):

    def __init__(self):
        super(CharbonnierPenalty, self).__init__()

    def forward(self, x, epsilon=1e-3, alpha=0.4):
        return (x ** 2 + epsilon ** 2) ** alpha


class PixelwiseReconstructionLoss(nn.Module):
    charbonnier_penalty = CharbonnierPenalty()

    def __init__(self):
        super(PixelwiseReconstructionLoss, self).__init__()

    def forward(self, I1, I2):
        diff = I1 - I2

        return  self.charbonnier_penalty(diff)


class SmoothnessLoss(nn.Module):
    charbonnier_penalty = CharbonnierPenalty()

    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, flow: torch.Tensor):
        flow_x = flow[:, 0::2, :, :]
        flow_y = flow[:, 1::2, :, :]

        delta_Vx_x = torch.gradient(flow_x, dim=1)[0]
        delta_Vx_y = torch.gradient(flow_x, dim=2)[0]
        delta_Vy_x = torch.gradient(flow_y, dim=1)[0]
        delta_Vy_y = torch.gradient(flow_y, dim=2)[0]

        return (self.charbonnier_penalty(delta_Vx_x) + self.charbonnier_penalty(delta_Vx_y) +
                self.charbonnier_penalty(delta_Vy_x) + self.charbonnier_penalty(delta_Vy_y))


class SSIMLoss(nn.Module):

    def __init__(self, patch=8, c1=1e-4, c2=1e-3):
        super(SSIMLoss, self).__init__()
        self.patch_size = patch
        self.c1 = c1
        self.c2 = c2

    def compute_ssim(self, patch1, patch2):
        mu1, mu2 = patch1.mean(), patch2.mean()
        sigma1, sigma2 = patch1.var(), patch2.var()
        sigma12 = ((patch1 - mu1) * (patch2 - mu2)).mean()
        numerator = (2 * mu1 * mu2 + self.c1) * (2 * sigma12 + self.c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + self.c1) * (sigma1 + sigma2 + self.c2)

        return numerator / denominator

    def forward(self, I1, I2):
        local_patch_size = self.patch_size

        if(I1.shape[2] < local_patch_size):
            local_patch_size = I1.shape[2]

        I1_patches = I1.unfold(2, local_patch_size, local_patch_size).unfold(3, local_patch_size, local_patch_size)
        I2_patches = I2.unfold(2, local_patch_size, local_patch_size).unfold(3, local_patch_size, local_patch_size)

        B, C, PH, PW, H, W = I1_patches.shape

        I1_patches = I1_patches.reshape(B, C, PH * PW, H, W)
        I2_patches = I2_patches.reshape(B, C, PH * PW, H, W)

        score = torch.stack([
            torch.tensor([
                self.compute_ssim(p1, p2) for p1, p2 in zip(I1_patches[b], I2_patches[b])
            ], device=I1.device) for b in range(B)
        ])

        return score


class MotionNetLoss(nn.Module):
    warp_transformation = WarpTransformation()

    SSIMLoss = SSIMLoss()
    SmoothnessLoss = SmoothnessLoss()
    PixelwiseLoss = PixelwiseReconstructionLoss()

    def __init__(self, weights=[1, 0.01, 1]):
        super(MotionNetLoss, self).__init__()

        self.weights = weights

    def forward(self, images, flows):
        _, _, FH, FW = flows.shape

        images_downsampled = F.interpolate(images, size=(FH, FW), mode='bilinear', align_corners=True)
        images_split = images_downsampled.split(3, dim=1)
        flow_split = flows.split(2, dim=1)

        images_warped = [self.warp_transformation(image, flow) for image, flow in zip(images_split[1:], flow_split)]
        images_warped = torch.cat(images_warped, dim=1)

        similarity_loss = self.SSIMLoss(images_downsampled[:, 0:30, :, :], images_warped).mean()
        smooth_loss = self.SmoothnessLoss(flows).mean()
        pixel_loss = self.PixelwiseLoss(images_downsampled[:, 0:30, :, :], images_warped).mean()

        return self.weights[0] * pixel_loss + self.weights[1] * smooth_loss + self.weights[2] * similarity_loss
