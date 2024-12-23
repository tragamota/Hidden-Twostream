import torch
import torch.nn.functional as F
from torch import nn


class WarpTransformation(torch.nn.Module):
    def __init__(self):
        super(WarpTransformation, self).__init__()

    def forward(self, image, flow):
        batch_size, channels, height, width = image.size()

        grid_x, grid_y = torch.meshgrid(torch.arange(0, width), torch.arange(0, height))

        grid_x = grid_x.float().to(image.device)
        grid_y = grid_y.float().to(image.device)

        grid_x = 2.0 * grid_x / (width - 1) - 1.0
        grid_y = 2.0 * grid_y / (height - 1) - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        grid = grid.repeat(batch_size, 1, 1, 1)

        flow = flow.permute(0, 2, 3, 1)
        grid = grid + flow

        # Rescale to [-1, 1] range (required by grid_sample)
        grid = grid * 2.0 - 1.0

        # Use grid_sample to warp the image based on the flow
        warped_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)

        return warped_image

