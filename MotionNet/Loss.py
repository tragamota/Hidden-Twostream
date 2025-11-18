import torch
import torch.nn.functional as F

from torch import nn
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

        return self.charbonnier_penalty(diff, alpha=0.4)


class SmoothnessLoss(nn.Module):
    charbonnier_penalty = CharbonnierPenalty()

    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, flow: torch.Tensor, border_mask: torch.Tensor):
        flow_x = flow[:, 0::2, :, :]
        flow_y = flow[:, 1::2, :, :]

        du_dx = flow_x[..., :, 1:] - flow_x[..., :, :-1]
        du_dy = flow_x[..., 1:, :] - flow_x[..., :-1, :]

        dv_dx = flow_y[..., :, 1:] - flow_y[..., :, :-1]
        dv_dy = flow_y[..., 1:, :] - flow_y[..., :-1, :]

        du_dx = F.pad(du_dx, (0, 1, 0, 0))
        du_dy = F.pad(du_dy, (0, 0, 0, 1))
        dv_dx = F.pad(dv_dx, (0, 1, 0, 0))
        dv_dy = F.pad(dv_dy, (0, 0, 0, 1))

        loss = (
            self.charbonnier_penalty(du_dx, alpha=0.3) +
            self.charbonnier_penalty(du_dy, alpha=0.3) +
            self.charbonnier_penalty(dv_dx, alpha=0.3) +
            self.charbonnier_penalty(dv_dy, alpha=0.3)
        )

        return loss.mean()


class SSIMLoss(nn.Module):
    def __init__(self, patch=8, c1=1e-4, c2=1e-3):
        super().__init__()
        self.patch = patch
        self.c1 = c1
        self.c2 = c2

    def forward(self, I1, I2):
        K = self.patch

        # Patch-wise means
        mu1 = F.avg_pool2d(I1, kernel_size=K, stride=K, padding=0)
        mu2 = F.avg_pool2d(I2, kernel_size=K, stride=K, padding=0)

        # Patch-wise variances
        sigma1 = F.avg_pool2d(I1 * I1, kernel_size=K, stride=K) - mu1 * mu1
        sigma2 = F.avg_pool2d(I2 * I2, kernel_size=K, stride=K) - mu2 * mu2

        # Patch-wise covariance
        sigma12 = F.avg_pool2d(I1 * I2, kernel_size=K, stride=K) - mu1 * mu2

        # SSIM numerator & denominator (per patch)
        numerator   = (2 * mu1 * mu2 + self.c1) * (2 * sigma12 + self.c2)
        denominator = (mu1 * mu1 + mu2 * mu2 + self.c1) * (sigma1 + sigma2 + self.c2)

        ssim = numerator / denominator

        return (1 - ssim).mean()



class MotionNetLoss(nn.Module):
    warp_transformation = WarpTransformation()

    SSIMLoss = SSIMLoss()
    SmoothnessLoss = SmoothnessLoss()
    PixelwiseLoss = PixelwiseReconstructionLoss()

    def __init__(self):
        super(MotionNetLoss, self).__init__()

    def forward(self, images, flows, smooth_weight, flow_scaling, border_mask, flow_mask):
        _, _, FH, FW = flows.shape

        images_downsampled = F.interpolate(images, size=(FH, FW), mode='bilinear', align_corners=True)
        images_split = images_downsampled.split(3, dim=1)
        flow_split = flows.split(2, dim=1)

        images_warped = [self.warp_transformation(image, flow) for image, flow in zip(images_split[1:], flow_split)]
        images_warped = torch.cat(images_warped, dim=1)

        similarity_loss = self.SSIMLoss(images_downsampled[:, 0:30, :, :], images_warped)
        smooth_loss = self.SmoothnessLoss(flows, flow_mask)
        pixel_loss = self.PixelwiseLoss(images_downsampled[:, 0:30, :, :], images_warped).mean()

        return pixel_loss + smooth_weight * smooth_loss + similarity_loss
