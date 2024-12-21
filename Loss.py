import torch
from torch import nn


class CharbonnierPenalty(nn.Module):

    def __init__(self):
        super(CharbonnierPenalty, self).__init__()

    def forward(self, x, epsilon=1e-3, alpha=0.4):
        return (x ** 2 + epsilon ** 2) ** alpha


class PixelwiseReconstructionLoss(nn.Module):
    charbonnier_penalty = CharbonnierPenalty()

    def __init__(self):
        super(PixelwiseReconstructionLoss, self).__init__()

    def forward(self, I1, I2, Vx, Vy):
        B, C, H, W = I1.shape

        x = torch.linspace(-1, 1, W, device=I1.device)
        y = torch.linspace(-1, 1, H, device=I1.device)

        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)

        flow = torch.stack((Vx, Vy), dim=-1)
        grid = grid + flow

        warped_I2 = torch.grid_sample(I2, grid, mode='bilinear', align_corners=False)

        diff = I1 - warped_I2

        return self.charbonnier_penalty(diff, self.epsilon, self.alpha).mean()


class SmoothnessLoss(nn.Module):
    charbonnier_penalty = CharbonnierPenalty()

    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, flow_x: torch.Tensor, flow_y: torch.Tensor):
        delta_Vx_x = torch.gradient(flow_x, dim=1)
        delta_Vx_y = torch.gradient(flow_x, dim=2)
        delta_Vy_x = torch.gradient(flow_y, dim=1)
        delta_Vy_y = torch.gradient(flow_y, dim=2)

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
        I1_patches = I1.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        I2_patches = I2.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

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
    SSIMLoss = SSIMLoss()

    def __init__(self, loss_weights=[]):
        super(MotionNetLoss, self).__init__()

    def forward(self):

        return 0
