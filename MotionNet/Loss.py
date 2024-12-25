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

    def forward(self, I1, I2, border_mask):
        diff = I1 - I2

        diff *= border_mask

        return self.charbonnier_penalty(diff, alpha=0.4)


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

        return (self.charbonnier_penalty(delta_Vx_x, alpha=0.3) + self.charbonnier_penalty(delta_Vx_y, alpha=0.3) +
                self.charbonnier_penalty(delta_Vy_x, alpha=0.3) + self.charbonnier_penalty(delta_Vy_y, alpha=0.3))


class SSIMLoss(nn.Module):
    def __init__(self, patch=8, c1=1e-4, c2=1e-3):
        super(SSIMLoss, self).__init__()
        self.patch_size = patch
        self.c1 = c1
        self.c2 = c2

    def compute_ssim(self, I1_patches, I2_patches):
        # Compute mean
        mu1 = I1_patches.mean(dim=(-1, -2), keepdim=True)  # (B, C, N, 1, 1)
        mu2 = I2_patches.mean(dim=(-1, -2), keepdim=True)

        # Compute variance
        sigma1 = ((I1_patches - mu1) ** 2).mean(dim=(-1, -2), keepdim=True)
        sigma2 = ((I2_patches - mu2) ** 2).mean(dim=(-1, -2), keepdim=True)

        # Compute covariance
        sigma12 = ((I1_patches - mu1) * (I2_patches - mu2)).mean(dim=(-1, -2), keepdim=True)

        # Compute SSIM
        numerator = (2 * mu1 * mu2 + self.c1) * (2 * sigma12 + self.c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + self.c1) * (sigma1 + sigma2 + self.c2)
        ssim = numerator / denominator

        return ssim  # (B, C, N, 1, 1)

    def forward(self, I1, I2):
        local_patch_size = min(self.patch_size, I1.shape[2], I1.shape[3])

        I1_patches = I1.unfold(2, local_patch_size, local_patch_size).unfold(3, local_patch_size, local_patch_size)
        I2_patches = I2.unfold(2, local_patch_size, local_patch_size).unfold(3, local_patch_size, local_patch_size)

        B, C, H_p, W_p, patch_h, patch_w = I1_patches.shape
        I1_patches = I1_patches.reshape(B, C, -1, patch_h, patch_w)  # (B, C, N, h, w)
        I2_patches = I2_patches.reshape(B, C, -1, patch_h, patch_w)

        ssim = self.compute_ssim(I1_patches, I2_patches)  # (B, C, N, 1, 1)

        loss = (1 - ssim)

        return loss.mean()


class MotionNetLoss(nn.Module):
    warp_transformation = WarpTransformation()

    SSIMLoss = SSIMLoss()
    SmoothnessLoss = SmoothnessLoss()
    PixelwiseLoss = PixelwiseReconstructionLoss()

    def __init__(self, weights=(1, 0.01, 1)):
        super(MotionNetLoss, self).__init__()

        self.weights = weights

    def forward(self, images, flows, smooth_weight, flow_scaling, border_mask):
        _, _, FH, FW = flows.shape

        images_downsampled = F.interpolate(images, size=(FH, FW), mode='bilinear', align_corners=True)
        images_split = images_downsampled.split(3, dim=1)
        flow_split = flows.split(2, dim=1)

        images_warped = [self.warp_transformation(image, flow_scaling * flow) for image, flow in zip(images_split[1:], flow_split)]
        images_warped = torch.cat(images_warped, dim=1)

        similarity_loss = self.SSIMLoss(images_downsampled[:, 0:30, :, :], images_warped)
        smooth_loss = self.SmoothnessLoss(flows).mean()
        pixel_loss = self.PixelwiseLoss(images_downsampled[:, 0:30, :, :], images_warped, border_mask).mean()

        return pixel_loss + smooth_weight * smooth_loss + similarity_loss
