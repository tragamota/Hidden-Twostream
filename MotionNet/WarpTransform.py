import torch
import torch.nn.functional as F

class WarpTransformation(torch.nn.Module):
    def __init__(self):
        super(WarpTransformation, self).__init__()

    def forward(self, image, flow):
        B, C, H, W = image.shape

        y, x = torch.meshgrid(
            torch.arange(0, H, device=image.device),
            torch.arange(0, W, device=image.device),
            indexing='ij'
        )

        x = 2.0 * x.float() / (W - 1) - 1.0
        y = 2.0 * y.float() / (H - 1) - 1.0

        base_grid = torch.stack((x, y), dim=-1)   # H × W × 2
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)

        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0, :, :] = 2.0 * flow[:, 0, :, :] / (W - 1)
        flow_norm[:, 1, :, :] = 2.0 * flow[:, 1, :, :] / (H - 1)

        flow_norm = flow_norm.permute(0, 2, 3, 1)

        sampling_grid = base_grid + flow_norm

        warped_image = F.grid_sample(
            image,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )

        return warped_image
